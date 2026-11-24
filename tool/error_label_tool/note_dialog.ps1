Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

$old = ''
if ($env:LABEL_NOTE_B64) {
    $old = [System.Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($env:LABEL_NOTE_B64))
}

$dialog = New-Object System.Windows.Forms.Form
$dialog.Text = 'Ghi chu video'
$dialog.Size = New-Object System.Drawing.Size(520, 280)
$dialog.StartPosition = 'CenterScreen'
$dialog.TopMost = $true
$dialog.MinimizeBox = $false
$dialog.MaximizeBox = $false

$description = New-Object System.Windows.Forms.Label
$description.Text = "Video $($env:LABEL_VIDEO)`nGhi chu (Ctrl+Enter hoac Luu de luu):"
$description.AutoSize = $true
$description.Location = New-Object System.Drawing.Point(16, 14)
$dialog.Controls.Add($description)

$inputBox = New-Object System.Windows.Forms.TextBox
$inputBox.Multiline = $true
$inputBox.ScrollBars = 'Vertical'
$inputBox.Location = New-Object System.Drawing.Point(18, 56)
$inputBox.Size = New-Object System.Drawing.Size(468, 120)
$inputBox.Anchor = 'Top, Bottom, Left, Right'
$inputBox.Text = $old
$inputBox.Add_KeyDown({ if ($_.Control -and $_.KeyCode -eq 'Enter') { $dialog.DialogResult = [System.Windows.Forms.DialogResult]::OK; $dialog.Close() } })
$dialog.Controls.Add($inputBox)

$save = New-Object System.Windows.Forms.Button
$save.Text = 'Luu'
$save.Location = New-Object System.Drawing.Point(310, 192)
$save.Anchor = 'Bottom, Right'
$save.DialogResult = [System.Windows.Forms.DialogResult]::OK
$dialog.Controls.Add($save)

$cancel = New-Object System.Windows.Forms.Button
$cancel.Text = 'Huy'
$cancel.Location = New-Object System.Drawing.Point(400, 192)
$cancel.Anchor = 'Bottom, Right'
$cancel.DialogResult = [System.Windows.Forms.DialogResult]::Cancel
$dialog.Controls.Add($cancel)
$dialog.CancelButton = $cancel

$dialog.Add_Shown({ $inputBox.Focus(); $inputBox.SelectionStart = $inputBox.Text.Length })
try {
    $result = $dialog.ShowDialog()
    if ($result -eq [System.Windows.Forms.DialogResult]::OK) {
        $bytes = [System.Text.Encoding]::UTF8.GetBytes($inputBox.Text)
        [Console]::WriteLine('SAVE:' + [Convert]::ToBase64String($bytes))
    } else {
        [Console]::WriteLine('CANCEL')
    }
} finally {
    $dialog.Dispose()
}
