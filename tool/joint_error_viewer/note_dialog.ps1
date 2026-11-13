Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

$dialog = New-Object System.Windows.Forms.Form
$dialog.Text = 'Ghi nhan loi'
$dialog.Size = New-Object System.Drawing.Size(480, 260)
$dialog.StartPosition = 'CenterScreen'
$dialog.TopMost = $true
$dialog.MinimizeBox = $false
$dialog.MaximizeBox = $false

$description = New-Object System.Windows.Forms.Label
$description.Text = "$($env:JOINT_REVIEW_LABEL) - frame $($env:JOINT_REVIEW_FRAME)`nGhi chu (co the de trong):"
$description.AutoSize = $true
$description.Location = New-Object System.Drawing.Point(16, 16)
$dialog.Controls.Add($description)

$inputBox = New-Object System.Windows.Forms.TextBox
$inputBox.Multiline = $true
$inputBox.ScrollBars = 'Vertical'
$inputBox.Location = New-Object System.Drawing.Point(18, 58)
$inputBox.Size = New-Object System.Drawing.Size(428, 100)
$inputBox.Anchor = 'Top, Bottom, Left, Right'
$dialog.Controls.Add($inputBox)

$save = New-Object System.Windows.Forms.Button
$save.Text = 'Luu'
$save.Location = New-Object System.Drawing.Point(270, 174)
$save.Anchor = 'Bottom, Right'
$save.DialogResult = [System.Windows.Forms.DialogResult]::OK
$dialog.Controls.Add($save)
$dialog.AcceptButton = $save

$cancel = New-Object System.Windows.Forms.Button
$cancel.Text = 'Huy'
$cancel.Location = New-Object System.Drawing.Point(360, 174)
$cancel.Anchor = 'Bottom, Right'
$cancel.DialogResult = [System.Windows.Forms.DialogResult]::Cancel
$dialog.Controls.Add($cancel)
$dialog.CancelButton = $cancel

$dialog.Add_Shown({ $inputBox.Focus() })
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
