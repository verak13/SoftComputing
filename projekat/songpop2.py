
from projekat.detect_record_module import real_time_detection
from pyautogui import *
import cv2
import numpy as np
import threading


def thread_record():
    real_time_detection("./MODEL.1LARGE1.h5", single_color=True, start_delay=3)

def recognize(path):

    template = cv2.imread(path)
    large_image = screenshot()
    dirname = os.path.dirname(__file__)
    large_image.save(os.path.join(dirname, 'slicice\slika.png'))
    img_rgb =  cv2.imread('slicice\slika.png')
    w, h = template.shape[:-1]

    res = cv2.matchTemplate(img_rgb, template, cv2.TM_CCOEFF_NORMED)
    threshold = .8
    loc = np.where(res >= threshold)
    location = []
    for pt in zip(*loc[::-1]):
        if not location:
            location.append(pt[0])
            location.append(pt[1])
        cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

    cv2.imwrite('result.png', img_rgb)

    if location:
        location[0] += h/2
        location[1] += w/2
    return location

if __name__ == '__main__':


    threading.Thread(target = thread_record).start()
    res = recognize("slicice/songpop2.PNG")
    if res:
        click(res)

    while(True):
        error = False
        sleep(2)
        xDugme = recognize("slicice/xDugme.PNG")
        if xDugme:
            click(xDugme)

        xDugme2 = recognize("slicice/xDugme2.PNG")
        if xDugme2:
            click(xDugme2)
        sleep(3)
        newGame = recognize("slicice/newGame.PNG")
        if newGame:
            print(newGame)
            click(newGame)
        while True:
            sleep(1)
            random = recognize("slicice/rng.PNG")
            if random:
                click(random)
            sleep(2)
            playlist = recognize("slicice/todaysHits.PNG")
            if playlist:
                click(playlist)
                sleep(8)
                xDugme2 = recognize("slicice/xDugme2.PNG")
                if xDugme2:
                    click(xDugme2)
                tooMuchGames = recognize("slicice/GameError.PNG")
                if tooMuchGames:
                    ok = recognize("slicice/ok.PNG")
                    click(ok)
                    error = True
                break
            else:
                back = recognize("slicice/nazad.PNG")
                if back:
                    click(back)
        if error:
            continue

        smer = 1
        while True:
            home = recognize("slicice/Home.PNG")
            if home:
                click(home)
                break
            else:
                if smer == 1:
                    smer = 0
                    moveRel(-100, -100)
                else:
                    smer = 1
                    moveRel(100, 100)
                sleep(3)


