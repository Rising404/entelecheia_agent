Option Explicit

Dim shell, fso, scriptDir, launcherPath, desktopPath, shortcutPath
Dim wscriptPath, electronPath, shortcut, errorNumber, errorDescription

Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)
launcherPath = fso.BuildPath(scriptDir, "start-electron-hidden.vbs")
If Not fso.FileExists(launcherPath) Then
  MsgBox "找不到隐藏启动器：" & launcherPath, vbCritical, "无法创建快捷方式"
  WScript.Quit 1
End If

desktopPath = shell.SpecialFolders("Desktop")
shortcutPath = fso.BuildPath(desktopPath, "Entelecheia.lnk")
wscriptPath = shell.ExpandEnvironmentStrings("%SystemRoot%\System32\wscript.exe")
electronPath = fso.BuildPath(scriptDir, "node_modules\electron\dist\electron.exe")

On Error Resume Next
Set shortcut = shell.CreateShortcut(shortcutPath)
shortcut.TargetPath = wscriptPath
shortcut.Arguments = QuoteArgument(launcherPath)
shortcut.WorkingDirectory = scriptDir
shortcut.Description = "启动或唤醒 Entelecheia"
shortcut.WindowStyle = 7
shortcut.Hotkey = "CTRL+SHIFT+SPACE"
If fso.FileExists(electronPath) Then
  shortcut.IconLocation = electronPath & ",0"
End If
shortcut.Save
errorNumber = Err.Number
errorDescription = Err.Description
On Error GoTo 0

If errorNumber <> 0 Then
  MsgBox "创建桌面快捷方式失败：" & errorDescription, vbCritical, "无法创建快捷方式"
  WScript.Quit 1
End If

MsgBox "桌面快捷方式已就绪。" & vbCrLf & _
  "双击它，或按 Ctrl+Shift+Space，即可启动/唤醒应用。", _
  vbInformation, "Entelecheia"
WScript.Quit 0

Function QuoteArgument(ByVal value)
  QuoteArgument = Chr(34) & Replace(CStr(value), Chr(34), Chr(34) & Chr(34)) & Chr(34)
End Function
