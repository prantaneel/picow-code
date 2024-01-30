import network, time
wifi_ssid = "neel"
wifi_password = "123456789"
wlan = network.WLAN(network.STA_IF)
wlan.active(True)
# wlan.config(pm = 0xa11140) # Diable powersave mode
wlan.disconnect()
time.sleep(5)
wlan.connect(wifi_ssid, wifi_password)

while not wlan.isconnected():
    print("Connecting to WiFi...")
    time.sleep(1)

print("Connected!")
print(wlan.ifconfig())
print("Refreshing...")
time.sleep(5)
print("Done")
