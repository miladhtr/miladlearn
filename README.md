# MiladLearn

📌 یک پکیج ساده برای اضافه کردن الگوریتم‌های سفارشی یادگیری ماشین.  
اولین الگوریتم: **MiladClassifier** (نسخه‌ی توسعه یافته از RandomForest).

---

## نصب
```bash
pip install git+https://github.com/YourUsername/miladlearn.git
```

## استفاده
```python
from miladlearn.ensemble import MiladClassifier

X = [[1,2,3],[2,4,6],[3,1,2],[4,2,1]]
y = [0,1,0,1]

clf = MiladClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)
print(clf.predict(X))
```

---

## تست
```bash
pytest tests/
```
