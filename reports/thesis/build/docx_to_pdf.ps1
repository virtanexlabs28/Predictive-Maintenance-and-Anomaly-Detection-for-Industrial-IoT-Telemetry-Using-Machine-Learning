# Converts a Word file to PDF with Microsoft Word (used to check the layout).
# Usage: powershell -File docx_to_pdf.ps1 -Source chapter.docx -Target chapter.pdf
param(
    [Parameter(Mandatory = $true)][string]$Source,
    [Parameter(Mandatory = $true)][string]$Target
)

$sourcePath = (Resolve-Path $Source).Path
$targetPath = [System.IO.Path]::GetFullPath($Target)
$word = New-Object -ComObject Word.Application
$word.Visible = $false
try {
    $document = $word.Documents.Open($sourcePath, $false, $true)   # read-only
    $document.ExportAsFixedFormat($targetPath, 17)                  # 17 = PDF
    Write-Output ("pages: " + $document.ComputeStatistics(2))
    Write-Output ("words: " + $document.ComputeStatistics(0))
    $document.Close($false)
}
finally {
    $word.Quit()
}
