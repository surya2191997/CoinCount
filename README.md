# CoinCount
Android App for counting total amount of coins

DEPENDENCIES

1. OpenCV 3.0
2. Numpy
3. Tensorflow

FILES

-PreProcess.cpp : Segments out the coins using HSV / Hough Circle Algorithm. The bounding rectangle of the coin is passed as an image to the pre-trained softmax classifier function .Total value of coins is returned and written to examples.txt

-CoinClassifier.py : A Softmax Regressor written in tensorflow to classify coin image into 4 classes (1,2,5,10). Weights and Biases thus obtained are written as a csv files which is used by PreProcess.cpp

-ServerScript.php : PHP script which is executed when android client sends an image to the server(stored in the upload folder). The script executes .exe file of PreProcess.cpp with the image sent as an arguement in command line. The output(Total coin value) is written to the examples.txt file. Android client can read the value from this file which is the output.

-test.html : HTML file to test the working of PHP script.

-ClientApp : Coin Count Android app which contains the client side code.




DIRECTIONS FOR SETTING UP SERVER ON LOCAL MACHINE

1. Install LAMP for ubuntu -- PHP,MySQL,Apache2;
2. Replace default-site(index.html) in the apache2 server and restart service.
3. Change permissions etc. (for /home/user/public_html/)
4. Open localhost/computer's IP in browser.







