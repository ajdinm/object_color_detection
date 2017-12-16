import cv2

vc = cv2.VideoCapture('/dev/video1')

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
    readh
jjjjjjj1Gj:wq
git add .
git status
git cv2om-m "reformat"
git push
vim srednja_xau
:wq
vim ~/.vimrc
$a
set colorcolum=80
else:
    rval = False

while rval:
    cv2.imshow("preview", frame)
    rval, frame = vc.read()
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break
