<?php
   if(isset($_FILES["uploadedfile"])){
      $errors= array();
      $file_name = $_FILES["uploadedfile"]["name"];
      $file_size =$_FILES["uploadedfile"]["size"];
      $file_tmp =$_FILES["uploadedfile"]["tmp_name"];
      $file_type=$_FILES["uploadedfile"]["type"];
      $file_ext=strtolower(end(explode('.',$_FILES["uploadedfile"]["name"])));
      
      echo $file_name;
      echo '<br/>';
      echo $file_size;
      echo '<br/>';
      echo $file_tmp;
      echo '<br/>';
      echo $file_type;
      echo '<br/>';
      echo $file_ext;
      echo '<br/>';
      echo sys_get_temp_dir();
      
      $expensions= array("jpeg","jpg","png");
      
      if(in_array($file_ext,$expensions)=== false){
         $errors[]="extension not allowed, please choose a JPEG or PNG file.";
      }
      
      if($file_size > 2097152){
         $errors[]='File size must be excately 2 MB';
      }
      
      if( move_uploaded_file($_FILES["uploadedfile"]["tmp_name"], "/home/surya/public_html/" . $_FILES["uploadedfile"]["name"])){
        
     
      
      $command = './app i.jpeg';
      exec($command);
      print_r($_SERVER);


      }else{
         
         print_r($errors);
      }
   }
?>



