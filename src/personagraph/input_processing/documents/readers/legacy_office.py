"""用于二进制 ``.doc`` 和 ``.ppt`` 源的显式 sandbox 桥梁。

二进制 Office 格式是原生 OOXML reader 无法安全解析的复合文档。只有当 Host 配置了精确的
LibreOffice 可执行文件，且本进程能将其置于 OS sandbox 后时，才会准入这些格式。转换器
只能看到源的私有副本、无法访问网络，并且只能写入全新临时目录；其 OOXML 输出随后还要
接受常规 DOCX/PPTX 容器验证。

此处不存在 PATH 发现或无 sandbox 回退。无法提供该边界的安装会继续返回
``unsupported_legacy_office``。
"""

from __future__ import annotations

import hashlib
import os
import platform
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ....configuration.app_settings import get_setting
from ...files import MAX_DOCUMENT_FILE_BYTES
from ..contracts import (
    DiagnosticCode,
    ProcessingDiagnostic,
    ProcessingResult,
    ProcessorFingerprint,
)
from .office import DOCX_READER, PPTX_READER, read_docx, read_pptx


LEGACY_OFFICE_SOFFICE_SETTING = "legacy_office_soffice"
LEGACY_OFFICE_CONVERSION_TIMEOUT_SECONDS = 60
_OLE_COMPOUND_FILE_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
_SUPPORTED_TARGETS = {".doc": ".docx", ".ppt": ".pptx"}


class LegacyOfficeConversionRunner(Protocol):
    """:class:`LegacyOfficeBridge` 使用的 Host 所有进程边界。"""

    def convert(
        self,
        *,
        soffice_path: Path,
        source_path: Path,
        output_dir: Path,
        profile_dir: Path,
        target_suffix: str,
        timeout_seconds: int,
    ) -> Path: ...


@dataclass(frozen=True, slots=True)
class MacOsSandboxExecLegacyOfficeRunner:
    """使用默认拒绝的 macOS sandbox profile 运行 LibreOffice。"""

    sandbox_exec_path: Path = Path("/usr/bin/sandbox-exec")

    def __post_init__(self) -> None:
        sandbox = self.sandbox_exec_path.expanduser()
        if sandbox.is_symlink():
            raise ValueError("sandbox-exec path must not be a symlink")
        sandbox = sandbox.resolve()
        if not sandbox.is_file() or not os.access(sandbox, os.X_OK):
            raise ValueError("sandbox-exec path must be an executable file")
        object.__setattr__(self, "sandbox_exec_path", sandbox)

    def convert(
        self,
        *,
        soffice_path: Path,
        source_path: Path,
        output_dir: Path,
        profile_dir: Path,
        target_suffix: str,
        timeout_seconds: int,
    ) -> Path:
        app_root = _libreoffice_read_root(soffice_path)
        workspace_root = source_path.parent.parent.resolve()
        profile_path = workspace_root / "legacy-office.sb"
        profile_path.write_text(
            _macos_sandbox_profile(
                workspace_root=workspace_root,
                libreoffice_root=app_root,
            ),
            encoding="utf-8",
        )
        conversion_filter = {
            ".docx": "docx:Office Open XML Text",
            ".pptx": "pptx:Impress MS PowerPoint 2007 XML",
        }[target_suffix]
        completed = subprocess.run(  # noqa: S603 - 固定参数，绝不使用 shell
            [
                str(self.sandbox_exec_path),
                "-f",
                str(profile_path),
                str(soffice_path),
                "--headless",
                "--safe-mode",
                "--nologo",
                "--nodefault",
                "--nolockcheck",
                "--norestore",
                "--nofirststartwizard",
                f"-env:UserInstallation={profile_dir.as_uri()}",
                "--convert-to",
                conversion_filter,
                "--outdir",
                str(output_dir),
                str(source_path),
            ],
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout_seconds,
            cwd=str(workspace_root),
            env={
                "HOME": str(profile_dir),
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
                "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
                "SAL_USE_VCLPLUGIN": "svp",
                "TMPDIR": str(workspace_root),
            },
        )
        converted = output_dir / f"{source_path.stem}{target_suffix}"
        if completed.returncode != 0:
            raise RuntimeError("sandboxed LibreOffice conversion failed")
        return converted


@dataclass(frozen=True, slots=True)
class LegacyOfficeBridge:
    """转换一个已准入的二进制 Office 源，随后使用原生 reader。"""

    soffice_path: Path
    runner: LegacyOfficeConversionRunner
    timeout_seconds: int = LEGACY_OFFICE_CONVERSION_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        supplied = self.soffice_path.expanduser()
        if supplied.is_symlink():
            raise ValueError("soffice path must not be a symlink")
        executable = supplied.resolve()
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise ValueError("soffice path must be an executable regular file")
        if not 1 <= self.timeout_seconds <= 300:
            raise ValueError("legacy Office timeout must be between 1 and 300 seconds")
        object.__setattr__(self, "soffice_path", executable)

    def processor_fingerprint(self, source_suffix: str) -> ProcessorFingerprint:
        target = _SUPPORTED_TARGETS.get(source_suffix.lower())
        if target is None:
            raise ValueError("legacy Office bridge only accepts .doc or .ppt")
        native = DOCX_READER if target == ".docx" else PPTX_READER
        executable_hash = _bounded_file_sha256(self.soffice_path)
        return ProcessorFingerprint(
            f"sandboxed-libreoffice-{source_suffix.lstrip('.')}+{native.reader}",
            f"1-{native.version}-{executable_hash[:16]}",
        )

    def read(self, path: Path) -> ProcessingResult:
        suffix = path.suffix.lower()
        target_suffix = _SUPPORTED_TARGETS.get(suffix)
        processor = self.processor_fingerprint(suffix)
        if target_suffix is None:  # pragma: no cover - 指纹检查会先行拒绝
            raise ValueError("legacy Office bridge only accepts .doc or .ppt")
        try:
            size = path.stat().st_size
            if size > MAX_DOCUMENT_FILE_BYTES:
                return _failure(
                    processor,
                    DiagnosticCode.LIMIT_REACHED,
                    "legacy Office source byte limit reached",
                )
            with path.open("rb") as source:
                if source.read(len(_OLE_COMPOUND_FILE_MAGIC)) != _OLE_COMPOUND_FILE_MAGIC:
                    return _failure(
                        processor,
                        DiagnosticCode.CORRUPT_SOURCE,
                        "legacy Office compound-file signature is invalid",
                    )
        except PermissionError:
            return _failure(
                processor,
                DiagnosticCode.PERMISSION_DENIED,
                "legacy Office source cannot be read",
            )
        except OSError:
            return _failure(
                processor,
                DiagnosticCode.CORRUPT_SOURCE,
                "legacy Office source metadata is unavailable",
            )

        try:
            with tempfile.TemporaryDirectory(
                prefix="personagraph-legacy-office-"
            ) as temporary:
                workspace = Path(temporary).resolve()
                source_dir = workspace / "source"
                output_dir = workspace / "output"
                profile_dir = workspace / "profile"
                source_dir.mkdir(mode=0o700)
                output_dir.mkdir(mode=0o700)
                profile_dir.mkdir(mode=0o700)
                staged_source = source_dir / f"source{suffix}"
                shutil.copyfile(path, staged_source)
                returned_path = self.runner.convert(
                    soffice_path=self.soffice_path,
                    source_path=staged_source,
                    output_dir=output_dir,
                    profile_dir=profile_dir,
                    target_suffix=target_suffix,
                    timeout_seconds=self.timeout_seconds,
                )
                if returned_path.is_symlink():
                    return _failure(
                        processor,
                        DiagnosticCode.CORRUPT_SOURCE,
                        "legacy Office converter returned a symbolic link",
                    )
                converted = returned_path.resolve()
                if (
                    converted.parent != output_dir
                    or converted.suffix.lower() != target_suffix
                    or not converted.is_file()
                ):
                    return _failure(
                        processor,
                        DiagnosticCode.CORRUPT_SOURCE,
                        "legacy Office converter returned an invalid output path",
                    )
                converted_size = converted.stat().st_size
                if converted_size <= 0 or converted_size > MAX_DOCUMENT_FILE_BYTES:
                    return _failure(
                        processor,
                        DiagnosticCode.LIMIT_REACHED,
                        "converted Office output is empty or exceeds the byte limit",
                    )
                native = read_docx(converted) if target_suffix == ".docx" else read_pptx(converted)
        except subprocess.TimeoutExpired:
            return _failure(
                processor,
                DiagnosticCode.LIMIT_REACHED,
                "legacy Office conversion time limit reached",
            )
        except PermissionError:
            return _failure(
                processor,
                DiagnosticCode.PERMISSION_DENIED,
                "legacy Office conversion was denied",
            )
        except (OSError, RuntimeError):
            return _failure(
                processor,
                DiagnosticCode.CORRUPT_SOURCE,
                "legacy Office conversion failed",
            )

        return ProcessingResult(
            elements=native.elements,
            processor=processor,
            diagnostics=(
                *native.diagnostics,
                ProcessingDiagnostic(
                    DiagnosticCode.FALLBACK_READER_USED,
                    detail=(
                        f"legacy {suffix} was converted to {target_suffix} "
                        "inside the configured sandbox"
                    ),
                ),
            ),
            page_manifest=native.page_manifest,
        )


def configured_legacy_office_bridge() -> LegacyOfficeBridge | None:
    """解析显式本地桥梁；绝不猜测 PATH，也绝不取消 sandbox。"""

    configured = get_setting(LEGACY_OFFICE_SOFFICE_SETTING)
    if not configured or platform.system() != "Darwin":
        return None
    try:
        return LegacyOfficeBridge(
            soffice_path=Path(configured),
            runner=MacOsSandboxExecLegacyOfficeRunner(),
        )
    except (OSError, ValueError):
        return None


def _failure(
    processor: ProcessorFingerprint,
    code: DiagnosticCode,
    detail: str,
) -> ProcessingResult:
    return ProcessingResult(
        elements=(),
        processor=processor,
        diagnostics=(ProcessingDiagnostic(code, detail=detail),),
    )


def _bounded_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            total += len(chunk)
            if total > MAX_DOCUMENT_FILE_BYTES:
                raise ValueError("configured soffice executable exceeds the hash bound")
            digest.update(chunk)
    return digest.hexdigest()


def _libreoffice_read_root(executable: Path) -> Path:
    for candidate in (executable, *executable.parents):
        if candidate.suffix.lower() == ".app":
            return candidate.resolve()
    return executable.parent.resolve()


def _sandbox_literal(path: Path) -> str:
    return str(path.resolve()).replace("\\", "\\\\").replace('"', '\\"')


def _macos_sandbox_profile(*, workspace_root: Path, libreoffice_root: Path) -> str:
    workspace = _sandbox_literal(workspace_root)
    office = _sandbox_literal(libreoffice_root)
    ancestor_reads = "\n".join(
        f'    (literal "{_sandbox_literal(parent)}")'
        for parent in reversed(workspace_root.resolve().parents)
    )
    return f"""(version 1)
(deny default)
(deny network*)
(allow process*)
(allow signal (target self))
(allow sysctl-read)
(allow mach-lookup
    (global-name \"com.apple.bsd.dirhelper\")
    (global-name \"com.apple.system.opendirectoryd.membership\"))
(allow file-read-metadata)
(allow file-read*
{ancestor_reads}
    (subpath \"/System\")
    (subpath \"/Library\")
    (subpath \"/usr\")
    (subpath \"/bin\")
    (subpath \"/sbin\")
    (subpath \"/private/etc\")
    (subpath \"/private/var/db\")
    (subpath \"/dev\")
    (subpath \"{office}\")
    (subpath \"{workspace}\"))
(allow file-write* (subpath \"{workspace}\"))
"""


__all__ = [
    "LEGACY_OFFICE_CONVERSION_TIMEOUT_SECONDS",
    "LEGACY_OFFICE_SOFFICE_SETTING",
    "LegacyOfficeBridge",
    "LegacyOfficeConversionRunner",
    "MacOsSandboxExecLegacyOfficeRunner",
    "configured_legacy_office_bridge",
]
