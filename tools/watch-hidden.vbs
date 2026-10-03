' Runs watch.cmd with a hidden window (0 = hidden, False = don't wait). Same pattern as n8n_pc launch-camera-hidden.vbs.
CreateObject("Wscript.Shell").Run "cmd /c """ & CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName) & "\watch.cmd""", 0, False
