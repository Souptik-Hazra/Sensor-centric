import os

f1 = r"c:\Users\User\Downloads\metr-la-dissertation-complete\EquiTrafficAI\frontend\src\modules\gis\MapLegend.jsx"
f2 = r"c:\Users\User\Downloads\metr-la-dissertation-complete\EquiTrafficAI\frontend\src\modules\gis\RouteControlPanel.jsx"

if os.path.exists(f1):
    os.remove(f1)
if os.path.exists(f2):
    os.remove(f2)

print("Temp components removed successfully.")
