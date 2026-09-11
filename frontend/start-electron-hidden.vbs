Option Explicit

Dim shell, fso, scriptDir, launcherScript, windowsRoot, powershellPath
Dim command, exitCode, logPath, errorNumber, errorDescription

Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)
launcherScript = fso.BuildPath(scriptDir, "start-electron.ps1")
If Not fso.FileExists(launcherScript) Then
  MsgBox "找不到启动脚本：" & launcherScript, vbCritical, "Entelecheia 无法启动"
  WScript.Quit 1
End If

windowsRoot = shell.ExpandEnvironmentStrings("%SystemRoot%")
powershellPath = fso.BuildPath(windowsRoot, "System32\WindowsPowerShell\v1.0\powershell.exe")
If Not fso.FileExists(powershellPath) Then
  powershellPath = "powershell.exe"
End If

command = QuoteArgument(powershellPath) & _
  " -NoLogo -NoProfile -NonInteractive -WindowStyle Hidden" & _
  " -ExecutionPolicy Bypass -File " & QuoteArgument(launcherScript) & _
  " -HiddenLauncher"

On Error Resume Next
exitCode = shell.Run(command, 0, True)
errorNumber = Err.Number
errorDescription = Err.Description
On Error GoTo 0

If errorNumber <> 0 Then
  MsgBox "无法启动 Windows PowerShell：" & errorDescription, vbCritical, "Entelecheia 无法启动"
  WScript.Quit 1
End If

If exitCode <> 0 Then
  logPath = shell.ExpandEnvironmentStrings("%LOCALAPPDATA%\Entelecheia\Logs\launcher.log")
  MsgBox "启动没有完成。请查看日志：" & vbCrLf & logPath, vbCritical, "Entelecheia 无法启动"
End If

WScript.Quit exitCode

Function QuoteArgument(ByVal value)
  QuoteArgument = Chr(34) & Replace(CStr(value), Chr(34), Chr(34) & Chr(34)) & Chr(34)
End Function
