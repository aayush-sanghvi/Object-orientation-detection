## GETTING STARTED
This project is used to detect the object, draw the contours and determine the angle of rotation between the object present in the test image and object in the template image.

## Prerequisites
OpenCV needs to be installed before executing the project.
You can use this link to install OpenCV.
<br>
[OpenCV Installation](https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html)

## INSTALLATION
1. Clone the repo
  ```sh
   git clone https://github.com/github_username/repo_name.git
   ```
2. Create build folder in the local repo and navigate to the newly created build folder
```sh
  mkdir build
  cd build
   ```
3. Execute cmake command
```sh
   cmake ..
   ```
4. Compile the code with the following command
```sh
   make
   ```
5. Execute the code
```sh
  ./mowito <i>path to the template image</i> <i>path to the test image</i>
   ```
