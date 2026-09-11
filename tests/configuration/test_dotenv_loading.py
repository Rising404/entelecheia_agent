import os

from personagraph.configuration import paths as paths_module


def test_default_dotenv_is_loaded_from_private_user_config(monkeypatch, tmp_path):
    key = "ENTELECHEIA_TEST_PRIVATE_DOTENV"
    private_config = tmp_path / "private-config"
    private_config.mkdir()
    (private_config / ".env").write_text(f"{key}=private-value\n", encoding="utf-8")

    monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(paths_module, "LOCAL_CONFIG_DIR", private_config)

    paths_module.load_dotenv()

    assert os.environ[key] == "private-value"


def test_explicit_dotenv_path_still_takes_priority(monkeypatch, tmp_path):
    key = "ENTELECHEIA_TEST_EXPLICIT_DOTENV"
    private_config = tmp_path / "private-config"
    private_config.mkdir()
    (private_config / ".env").write_text(f"{key}=default-value\n", encoding="utf-8")
    explicit_path = tmp_path / "controlled.env"
    explicit_path.write_text(f"{key}=explicit-value\n", encoding="utf-8")

    monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(paths_module, "LOCAL_CONFIG_DIR", private_config)

    paths_module.load_dotenv(explicit_path)

    assert os.environ[key] == "explicit-value"


def test_repository_or_working_directory_dotenv_is_not_loaded(monkeypatch, tmp_path):
    key = "ENTELECHEIA_TEST_REPOSITORY_DOTENV"
    private_config = tmp_path / "private-config"
    private_config.mkdir()
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / ".env").write_text(f"{key}=must-not-load\n", encoding="utf-8")

    monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(paths_module, "LOCAL_CONFIG_DIR", private_config)
    monkeypatch.chdir(checkout)

    paths_module.load_dotenv()

    assert key not in os.environ


def test_dotenv_does_not_override_existing_process_environment(monkeypatch, tmp_path):
    key = "ENTELECHEIA_TEST_EXISTING_DOTENV"
    explicit_path = tmp_path / "controlled.env"
    explicit_path.write_text(f"{key}=file-value\n", encoding="utf-8")
    monkeypatch.setenv(key, "process-value")

    paths_module.load_dotenv(explicit_path)

    assert os.environ[key] == "process-value"
