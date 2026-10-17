const crypto = require("node:crypto");
const fs = require("node:fs");
const path = require("node:path");

function buildRuntimeIdentity(root) {
  const canonicalRoot = fs.realpathSync.native(root);
  const packageRoot = path.join(canonicalRoot, "src", "personagraph");
  const files = fs.readdirSync(packageRoot, { recursive: true, withFileTypes: true })
    .filter((entry) => entry.isFile() && entry.name.endsWith(".py"))
    .map((entry) => path.join(entry.parentPath, entry.name));
  for (const name of ["pyproject.toml", "uv.lock"]) {
    const candidate = path.join(canonicalRoot, name);
    if (fs.existsSync(candidate)) files.push(candidate);
  }
  const relativePath = (file) => path.relative(canonicalRoot, file).split(path.sep).join("/");
  // 与 API 的 runtime_identity.py 共用字节协议；跨语言回归测试锁定算法一致性。
  files.sort((a, b) => Buffer.compare(Buffer.from(relativePath(a)), Buffer.from(relativePath(b))));
  const digest = crypto.createHash("sha256");
  for (const file of files) {
    digest.update(relativePath(file) + "\0", "utf8");
    digest.update(crypto.createHash("sha256").update(fs.readFileSync(file)).digest());
  }
  return {
    checkout_sha256: crypto.createHash("sha256")
      .update(canonicalRoot.split(path.sep).join("/"), "utf8").digest("hex"),
    source_sha256: digest.digest("hex")
  };
}

module.exports = { buildRuntimeIdentity };
