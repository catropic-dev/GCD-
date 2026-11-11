param([Parameter(Mandatory=$true)][string]$Job)
$ErrorActionPreference = 'Stop'
$clips = Get-Content -LiteralPath $Job -Raw -Encoding UTF8 | ConvertFrom-Json
$speaker = New-Object -ComObject SAPI.SpVoice
$voice = $speaker.GetVoices('Language=804') | Select-Object -First 1
if (!$voice) { throw 'Install a Simplified Chinese desktop voice or use --audio-dir.' }
$speaker.Voice = $voice
foreach ($clip in $clips) {
    $stream = New-Object -ComObject SAPI.SpFileStream
    $stream.Format.Type = 22
    $stream.Open($clip.path, 3, $false)
    try {
        $speaker.AudioOutputStream = $stream
        [void]$speaker.Speak($clip.text, 16)
    } finally {
        $stream.Close()
    }
}
