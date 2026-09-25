const { contextBridge, ipcRenderer } = require("electron");

function sidecarInfo() {
  const prefix = "--personagraph-sidecar=";
  const arg = process.argv.find((value) => value.startsWith(prefix));
  if (!arg) return null;
  try {
    return JSON.parse(decodeURIComponent(arg.slice(prefix.length)));
  } catch {
    return null;
  }
}

const sidecar = sidecarInfo();

contextBridge.exposeInMainWorld("personagraphDesktop", {
  apiBase: sidecar?.apiBase || "http://127.0.0.1:8765",
  platform: process.platform,
  shell: "electron",
  sidecar,
  version: process.versions.electron,
  chooseDocumentPath: () => ipcRenderer.invoke("personagraph:choose-document"),
  chooseDirectory: () => ipcRenderer.invoke("personagraph:choose-directory"),
  ensureApiSidecar: () => ipcRenderer.invoke("personagraph:ensure-api-sidecar"),
  onRuntimeReconciliation: (callback) => {
    if (typeof callback !== "function") return () => {};
    const listener = (_event, payload) => callback(payload);
    ipcRenderer.on("personagraph:runtime-reconcile", listener);
    return () => ipcRenderer.removeListener("personagraph:runtime-reconcile", listener);
  }
});
