import win32com
import win32con
import win32api
from time import sleep
import win32gui
import keyboard
import subprocess
import win32com.client

cmd = '"C:\\Program Files\\mRayClient\\bin\\mRayClient.exe" --hotkey'
p = subprocess.Popen(cmd)
pid = p.pid
input("Press Enter to continue...")
def windowEnumerationHandler(hwnd, top_windows):
    top_windows.append((hwnd, win32gui.GetWindowText(hwnd)))

if __name__ == "__main__":
    results = []
    top_windows = []
    win32gui.EnumWindows(windowEnumerationHandler, top_windows)
    for i in top_windows:
        if "mray client" in i[1].lower():
            print(i[1].lower())
            win32gui.ShowWindow(i[0], 5)
            win32gui.SetForegroundWindow(i[0])
            #win32gui.SetFocus(i[0])

            for j in range(0,10):
                keyboard.press_and_release('down')
                keyboard.press_and_release('down')
                keyboard.press_and_release('down')
                keyboard.press_and_release('down')
                keyboard.press_and_release('down')
                keyboard.press_and_release('down')
                keyboard.press_and_release('down')
                sleep(3)
                print("printing")
                keyboard.press_and_release('u')
                keyboard.press_and_release('u')
                keyboard.press_and_release('u')
                keyboard.press_and_release('u')
                keyboard.write('teeeeeest')
                temp = win32api.PostMessage(i[0], win32con.WM_CHAR, ord("x"), 0)
                print(temp)
                sleep(0.1)
                temp = win32api.PostMessage(i[0], win32con.WM_CHAR, ord("x"), 0)
                print(temp)
                sleep(0.1)
                temp = win32api.PostMessage(i[0], win32con.WM_CHAR, ord("x"), 0)
                print(temp)
                sleep(0.1)
                temp = win32api.PostMessage(i[0], win32con.WM_CHAR, ord("x"), 0)
                print(temp)
                sleep(0.1)

                win32api.keybd_event(win32con.VK_DOWN, 0, 0, 0)
                win32api.keybd_event(win32con.VK_DOWN, 0, win32con.KEYEVENTF_KEYUP, 0)
                win32api.keybd_event(win32con.VK_DOWN, 0, 0, 0)
                win32api.keybd_event(win32con.VK_DOWN, 0, win32con.KEYEVENTF_KEYUP, 0)
                win32api.keybd_event(win32con.VK_DOWN, 0, 0, 0)
                win32api.keybd_event(win32con.VK_DOWN, 0, win32con.KEYEVENTF_KEYUP, 0)

                sleep(0.1)
                temp = win32api.PostMessage(i[0], win32con.WM_KEYUP, win32con.VK_DOWN, 0)
                print(temp)
                sleep(0.1)
                temp = win32api.PostMessage(i[0], win32con.WM_KEYDOWN, win32con.VK_DOWN, 0)
                print(temp)
                sleep(0.1)
                temp = win32api.PostMessage(i[0], win32con.WM_KEYUP, win32con.VK_DOWN, 0)
                print(temp)
                sleep(0.1)
                temp = win32api.SendMessage(i[0], win32con.WM_SYSKEYDOWN, win32con.VK_DOWN, 0)
                print(temp)
                sleep(0.1)
                temp = win32api.SendMessage(i[0], win32con.WM_SYSKEYUP, win32con.VK_DOWN, 0)
                print(temp)
                temp = win32api.SendMessage(i[0], win32con.WM_SYSKEYDOWN, win32con.VK_DOWN, 0)
                print(temp)
                sleep(0.1)
                temp = win32api.PostMessage(i[0], win32con.WM_SYSKEYUP, win32con.VK_DOWN, 0)
                print(temp)
                temp = win32api.PostMessage(i[0], win32con.WM_SYSKEYDOWN, win32con.VK_DOWN, 0)
                print(temp)
                sleep(0.1)
                temp = win32api.SendMessage(i[0], win32con.WM_SYSKEYUP, win32con.VK_DOWN, 0)
                print(temp)
                temp = win32api.SendMessage(i[0], win32con.WM_SYSKEYDOWN, win32con.VK_DOWN, 0)
                print(temp)
                sleep(0.1)
                temp = win32api.SendMessage(i[0], win32con.WM_SYSKEYUP, win32con.VK_DOWN, 0)
                print(temp)

            break

#[hwnd] No matter what people tell you, this is the handle meaning unique ID,
#["Notepad"] This is the application main/parent name, an easy way to check for examples is in Task Manager
#["test - Notepad"] This is the application sub/child name, an easy way to check for examples is in Task Manager clicking dropdown arrow
#hwndMain = win32gui.FindWindow("Notepad", "test - Notepad") this returns the main/parent Unique ID
#hwndMain = win32gui.FindWindow("mRayClient.exe", "mRay Client 5.7.2")

#["hwndMain"] this is the main/parent Unique ID used to get the sub/child Unique ID
#[win32con.GW_CHILD] I havent tested it full, but this DOES get a sub/child Unique ID, if there are multiple you'd have too loop through it, or look for other documention, or i may edit this at some point ;)
#hwndChild = win32gui.GetWindow(hwndMain, win32con.GW_CHILD) this returns the sub/child Unique ID
#hwndChild = win32gui.GetWindow(hwndMain, win32con.GW_CHILD)

#print(hwndMain) #you can use this to see main/parent Unique ID
#print(hwndChild)  #you can use this to see sub/child Unique ID

#While(True) Will always run and continue to run indefinitely
#while(True):
    #[hwndChild] this is the Unique ID of the sub/child application/proccess
    #[win32con.WM_CHAR] This sets what PostMessage Expects for input theres KeyDown and KeyUp as well
    #[0x44] hex code for D
    #[0]No clue, good luck!
    #temp = win32api.PostMessage(hwndChild, win32con.WM_CHAR, 0x44, 0) returns key sent
    #temp = win32api.PostMessage(hwndChild, win32con.WM_CHAR, 0x09, 0)

    #print(temp) prints the returned value of temp, into the console
    #print(temp)
    #sleep(1) this waits 1 second before looping through again
    #sleep(1)