import cv2
import numpy as np

# 1. 讀取你的兩張真實照片 (支援繁體中文路徑與檔名)
def read_image_utf8(path):
    with open(path, "rb") as f:
        return cv2.imdecode(np.frombuffer(f.read(), dtype=np.uint8), cv2.IMREAD_COLOR)

# 💡 請確認 template.png 與 test.png 在你的資料夾中，或填入完整路徑
img_template = read_image_utf8("template.jpg")  # 左邊：乾淨範本圖
img_test = read_image_utf8("test.jpg")          # 右邊：有瑕疵/雜物測試圖

if img_template is None or img_test is None:
    print("❌ 讀取照片失敗，請確認圖片檔名與路徑是否正確！")
    exit()

# 轉灰階
gray_temp = cv2.cvtColor(img_template, cv2.COLOR_BGR2GRAY)
gray_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2GRAY)

# 2. ORB 特徵偵測 (設定最多找出 3000 個特徵點)
orb = cv2.ORB_create(nfeatures=3000)
kp_temp, des_temp = orb.detectAndCompute(gray_temp, None)
kp_test, des_test = orb.detectAndCompute(gray_test, None)

# 3. 特徵配對
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des_temp, des_test)
matches = sorted(matches, key=lambda x: x.distance)

# 4. 提取配對點座標，送入 RANSAC 篩選
pts_temp = np.float32([kp_temp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
pts_test = np.float32([kp_test[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

if len(matches) > 10:
    # 使用 RANSAC 自動過濾掉「交叉牽錯」的誤判點
    M, inliers_mask = cv2.findHomography(pts_test, pts_temp, cv2.RANSAC, 5.0)
    
    # 篩選出正確的配對點來繪製連線圖
    inliers_matches = [matches[i] for i in range(len(matches)) if inliers_mask[i][0] == 1]
    img_clean_matches = cv2.drawMatches(img_template, kp_temp, img_test, kp_test, inliers_matches[:50], None, flags=2)

    # 5. 空間透視變換 (Warp)：把測試圖拉正重合
    h, w, c = img_template.shape
    img_test_aligned = cv2.warpPerspective(img_test, M, (w, h))

    # 6. 影像相減與雜訊過濾
    gray_aligned = cv2.cvtColor(img_test_aligned, cv2.COLOR_BGR2GRAY)
    
    # 🌟【終極修正一】：全域明暗對齊 (Min-Max Normalization)
    # 將左邊灰階圖與右邊灰階圖的動態範圍，強行統一縮放到 0 ~ 255 之間
    # 這能讓「一亮一暗」的兩張圖，整體的灰階基調變得完全對等！
    gray_temp_norm = cv2.normalize(gray_temp, None, 0, 255, cv2.NORM_MINMAX)
    gray_aligned_norm = cv2.normalize(gray_aligned, None, 0, 255, cv2.NORM_MINMAX)
    
    # 再加上 CLAHE 局部校正，雙重保險
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_temp_enhanced = clahe.apply(gray_temp_norm)
    gray_aligned_enhanced = clahe.apply(gray_aligned_norm)
    
    # 模糊化（抹平微小光影細紋）
    gray_temp_blur = cv2.GaussianBlur(gray_temp_enhanced, (5, 5), 0)
    gray_aligned_blur = cv2.GaussianBlur(gray_aligned_enhanced, (5, 5), 0)
    
    # 計算絕對差值（這時候背景的像素差應該會降到極低）
    img_diff = cv2.absdiff(gray_temp_blur, gray_aligned_blur)
    
    # 🌟【終極修正二】：將二值化門檻（Threshold）大幅調高！
    # 因為你人為製造的亮度差很大，我們把門檻從 40 調高到 80 甚至 100。
    # 這樣一來，背景殘留的微小明暗差（小於 80）就會被當作黑色濾掉，只有電池盒這種「實體多出來、形狀完全不同」的地方才能突破 80 的大關！
    _, img_thresh = cv2.threshold(img_diff, 80, 255, cv2.THRESH_BINARY)
    
    # 形態學過濾
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    img_closed = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel_close)
    
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    img_cleaned_diff = cv2.morphologyEx(img_closed, cv2.MORPH_OPEN, kernel_open)
    
    # =========================================================
    # 🖥️ 視窗設定：方法一（建立可自由縮放、不爆螢幕的視窗）
    # =========================================================
    
    # ---- 視窗 1：特徵點連線 ----
    win_name1 = "1. Clean Matches (RANSAC Filtered)"
    cv2.namedWindow(win_name1, cv2.WINDOW_NORMAL)  # 設定為可縮放模式
    cv2.resizeWindow(win_name1, 900, 500)          # 預設寬 900, 高 500
    cv2.imshow(win_name1, img_clean_matches)

    # ---- 視窗 2：拉正後的測試圖 ----
    win_name2 = "2. Aligned Test Image"
    cv2.namedWindow(win_name2, cv2.WINDOW_NORMAL)  # 設定為可縮放模式
    cv2.resizeWindow(win_name2, 600, 450)          # 預設寬 600, 高 450
    cv2.imshow(win_name2, img_test_aligned)

    # ---- 視窗 3：抓出來的異常點(垃圾) ----
    win_name3 = "3. Detected Anomalies (Trash)"
    cv2.namedWindow(win_name3, cv2.WINDOW_NORMAL)  # 設定為可縮放模式
    cv2.resizeWindow(win_name3, 600, 450)          # 預設寬 600, 高 450
    cv2.imshow(win_name3, img_cleaned_diff)
    
    print("🎉 視窗已成功縮放！你現在可以用滑鼠自由拖曳、調整這三個視窗的大小。")
    print("👉 請按螢幕任意鍵關閉所有視窗。")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("❌ 特徵點配對數量不足，無法計算對齊矩陣！")