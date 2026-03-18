Set WshShell = CreateObject("WScript.Shell")

komut = "cmd.exe /c venv\Scripts\activate.bat && python interface.py || pause"

' Sondaki 0 rakami pencerenin GIZLI (Hidden) olmasini saglar.
WshShell.Run komut, 1, False

Set WshShell = Nothing