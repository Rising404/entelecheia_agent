const os = require("node:os");
const path = require("node:path");

const APPLICATION_DIRECTORY_NAME = "Entelecheia";

function nonEmpty(value) {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

function absoluteEnvironmentPath(env, name) {
  const candidate = nonEmpty(env[name]);
  return candidate && path.isAbsolute(candidate) ? candidate : null;
}

function explicitStoragePath(env, name) {
  const candidate = nonEmpty(env[name]);
  if (!candidate) return null;
  if (!path.isAbsolute(candidate)) {
    throw new Error(`${name} must be an absolute path`);
  }
  return candidate;
}

function pathIsInside(candidate, root) {
  if (!nonEmpty(root)) return false;
  const relative = path.relative(path.resolve(root), path.resolve(candidate));
  return relative === "" || (
    relative !== ".." &&
    !relative.startsWith(`..${path.sep}`) &&
    !path.isAbsolute(relative)
  );
}

function defaultStorageRoots({
  userDataPath,
  env,
  platform,
  homeDirectory
}) {
  const userData = nonEmpty(userDataPath);
  const home = nonEmpty(homeDirectory);

  if (platform === "darwin") {
    const applicationRoot = userData || (home && path.join(
      home,
      "Library",
      "Application Support",
      APPLICATION_DIRECTORY_NAME
    ));
    if (!applicationRoot) return null;
    return {
      stateDir: path.join(applicationRoot, "state"),
      configDir: path.join(applicationRoot, "config")
    };
  }

  if (platform === "win32") {
    const dataRoot = absoluteEnvironmentPath(env, "LOCALAPPDATA") ||
      (home && path.join(home, "AppData", "Local"));
    const configRoot = absoluteEnvironmentPath(env, "APPDATA") ||
      (home && path.join(home, "AppData", "Roaming"));
    if (dataRoot && configRoot) {
      return {
        stateDir: path.join(dataRoot, APPLICATION_DIRECTORY_NAME, "state"),
        configDir: path.join(configRoot, APPLICATION_DIRECTORY_NAME, "config")
      };
    }
  } else {
    const dataRoot = absoluteEnvironmentPath(env, "XDG_DATA_HOME") ||
      (home && path.join(home, ".local", "share"));
    const configRoot = absoluteEnvironmentPath(env, "XDG_CONFIG_HOME") ||
      (home && path.join(home, ".config"));
    if (dataRoot && configRoot) {
      return {
        stateDir: path.join(dataRoot, APPLICATION_DIRECTORY_NAME, "state"),
        configDir: path.join(configRoot, APPLICATION_DIRECTORY_NAME, "config")
      };
    }
  }

  if (!userData) return null;
  return {
    stateDir: path.join(userData, "state"),
    configDir: path.join(userData, "config")
  };
}

function resolveSidecarStorage({
  userDataPath,
  env = process.env,
  platform = process.platform,
  homeDirectory = os.homedir(),
  sourceRoot = null
} = {}) {
  const explicitStateDir = explicitStoragePath(env, "PERSONAGRAPH_STATE_DIR");
  const explicitConfigDir = explicitStoragePath(
    env,
    "PERSONAGRAPH_LOCAL_CONFIG_DIR"
  );
  const explicitTokenFile = explicitStoragePath(
    env,
    "PERSONAGRAPH_API_TOKEN_FILE"
  );
  const defaults = defaultStorageRoots({
    userDataPath,
    env,
    platform,
    homeDirectory
  });

  if ((!explicitStateDir || !explicitConfigDir) && !defaults) {
    throw new Error("A per-user storage root is required when sidecar overrides are absent");
  }

  const stateDir = explicitStateDir || defaults.stateDir;
  const configDir = explicitConfigDir || defaults.configDir;
  const apiTokenPath = explicitTokenFile || path.join(stateDir, "api_secret");

  for (const [name, candidate] of [
    ["PERSONAGRAPH_STATE_DIR", stateDir],
    ["PERSONAGRAPH_LOCAL_CONFIG_DIR", configDir],
    ["PERSONAGRAPH_API_TOKEN_FILE", apiTokenPath]
  ]) {
    if (pathIsInside(candidate, sourceRoot)) {
      throw new Error(`${name} must be outside the source checkout`);
    }
  }

  return { stateDir, configDir, apiTokenPath };
}

module.exports = { resolveSidecarStorage };
