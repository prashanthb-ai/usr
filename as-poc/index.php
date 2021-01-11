<?php
    if($_SERVER['REQUEST_METHOD'] == "POST" and isset($_POST['button1']))
    {
        button1();
    }
    if($_SERVER['REQUEST_METHOD'] == "POST" and isset($_POST['button2']))
    {
        button2();
    }
?>

<form method="post" action="button.html">
    <input type="submit" name="button1"
            class="button" value="Button1" />

    <input type="submit" name="button2"
            class="button" value="Button2" />
</form>
</head>

</html>
