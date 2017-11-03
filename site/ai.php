<?php

$name = md5(time()+rand());

$img = $_POST['img'];
$img = str_replace('data:image/jpeg;base64,', '', $img);
$img = str_replace(' ', '+', $img);
$img = base64_decode($img);
file_put_contents('imgs/'.$name.'.jpeg', $img);

$command = escapeshellcmd('python preprocess.py imgs/'.$name.'.jpeg');
shell_exec($command);

$command = escapeshellcmd('python controller.py imgs/'.$name.'.jpeg');
print_r(shell_exec($command));

?>