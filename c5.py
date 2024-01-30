import sys
sys.path.reverse()

import dht, time
import machine
import utime
import bme280_float as bme280
import network
from umqtt.robust import MQTTClient
import ujson
from device import node_index, node_id

STRATEGY = "DIFFUSION_ATC"
STATE = "HALT"
CLEAR = 0
NORM_FACTOR = 25
#<---------DEVICE CONFIG------------------------->
CLIENT_ID = bytes(node_id, "utf-8")
server_url = "efbeb8c27ac14a0cb2bf84cb91d64e0b.s2.eu.hivemq.cloud"
SERVER_URL = bytes(server_url, "utf-8")
PORT = 8883
broker_username = b'picow_0'
broker_password = b"Pi123456789"
connection_time = 7200
CENTRAL_TOPIC = "dashboard_socket"
p_id = "12"
SYNC_TOPIC = "picow_sync_{}".format(node_index)
#<------------STRATEGY PARAMS-------------------->
N = 3
w_prev = [0]*N
w_iter = [0]*N
psi_iter = [0]*N
psi_bool = 0
MAX_ITERATIONS = 100
mu = 0.001
#<---------TOPOLOGY DATA------------------------->
neighbour_index = [] #no specific order
edge_weight = [1]
M = len(neighbour_index)
next_iter_psi = [[]]*M
next_iter_w_prev = [[]]*M
n_map = {}
for _iter in range(M):
    n_map.update({neighbour_index[_iter]: _iter})
    
    
#<---------CONFIGURING THE SENSORS------------------------->
# Define GPIO pin 
sensor_power_pin = machine.Pin(4)  # Replace with the desired pin number

# Set GPIO pins as output
sensor_power_pin.init(mode=machine.Pin.OUT)

# Turn on power to sensors
sensor_power_pin.value(1)

# Add a delay if needed to ensure sensors have stable power
utime.sleep(2)

dht_pin = machine.Pin(2)
dt = dht.DHT11(dht_pin)

i2c = machine.I2C(0, sda=machine.Pin(8), scl=machine.Pin(9))  # Adjust GPIO pins as needed
bme = bme280.BME280(i2c=i2c)

def connectMQTT(callback_func):
    client = MQTTClient(client_id=CLIENT_ID,
                        server=SERVER_URL,
                        port=PORT,
                        user=broker_username,
                        password=broker_password,
                        keepalive=connection_time,
                        ssl=True,
                        ssl_params={'server_hostname': server_url}
                        )
    client.set_callback(callback_func)
    client.connect()
    print("{} connected to MQTT broker".format(CLIENT_ID))
    return client

#<---------DIFFUSUION EQUATION------------------------->
def dotpdt(u, v):
    n = len(u)
    m = len(v)
    assert n==m
    dp = 0
    for i in range(n):
      dp = dp + u[i]*v[i]
    return dp

def calculate_psi_atc_mse(d, u, mu):
    """
    u: the regressive vector of size 1xN
    d: the high cost input
    mu: learning rate
    """
    global w_iter
    global psi_iter
    grad = d - dotpdt(u, w_iter)
    for _iter in range(N):
        psi_iter[_iter] = w_iter[_iter] + 2*mu*grad*u[_iter]
    
    return
def calculate_w_updated_atc_mse(index_l, psi_l):
    global edge_weight
    global w_iter
    print("PSI Received: {}".format(psi_l))
    a_lk = edge_weight[index_l]
    for i in range(N):
        w_iter[i] = w_iter[i] + a_lk*psi_l[i]
    return
#<---------CONSENSUS EQUATION------------------------->
def calculate_psi_con_mse(index_l, w_prev_l):
    global edge_weight, psi_iter
    print("W_PREV Received: {}".format(w_prev_l))
    a_lk = edge_weight[index_l]
    for i in range(N):
        psi_iter[i] = psi_iter[i] + a_lk*w_prev_l[i]
    return

def calculate_w_updated_con_cta_mse(d, u, mu):
    global w_iter, psi_iter, w_prev
    if STRATEGY == "CONSENSUS":
        grad = d - dotpdt(u, w_prev)
    if STRATEGY == "DIFFUSION_CTA":
        grad = d - dotpdt(u, psi_iter)
    for _iter in range(N):
        w_prev[_iter] = w_iter[_iter]
        w_iter[_iter] = w_iter[_iter] + 2*mu*grad*u[_iter]
    return
#<---------SET CALLBACK------------------------->
def callback_function(topic, msg):
    global STRATEGY, psi_bool, STATE, neighbour_index, edge_weight, mu, MAX_ITERATIONS, w_prev, w_iter, psi_iter, psi_bool, n_map, N, M, p_id, CLEAR, next_iter_psi, next_iter_w_prev
    if msg==b'':
        return
    topic = ''.join(chr(b) for b in topic)
    msg_str = ''.join(chr(b) for b in msg)
    print(topic, msg)
    if topic == SYNC_TOPIC:
        payload = ujson.loads(msg_str)
        sync = payload["sync"]
        if sync == 1:
            STATE = "HALT"
            CLEAR = 1
        else:
            #initialisation
            if p_id == payload["p_id"]:
                return
            STATE = "TRAIN"
            STRATEGY = payload["strategy"]
            p_id = payload["p_id"]
            neighbour_index = payload["neighbours"]
            edge_weight = payload["edge_weights"]
            mu = payload["mu"]
            MAX_ITERATIONS = payload["iter"]
            N = payload["weightSize"]
            w_prev = [0]*N
            w_iter = [0]*N
            psi_iter = [0]*N
            psi_bool = 0
            M = len(neighbour_index)
            next_iter_psi = [[]]*M
            next_iter_w_prev = [[]]*M
            n_map = {}
            for _iter in range(M):
                n_map.update({neighbour_index[_iter]: _iter})
        return
    neighbour_payload = ujson.loads(msg_str)
    if neighbour_payload["p_id"] != p_id:
        return
    if STRATEGY == "DIFFUSION_ATC":
        
        psi_neighbour = neighbour_payload["psi_iter"]
        neighbour_ind = int(topic.split("_")[-1])
        psi_neighbour = [float(i) for i in psi_neighbour]
        n_index = n_map[neighbour_ind]
        has_got = (psi_bool >> n_index)&1
        if has_got or STATE=="HALT" :
            #check if we already have the value for this
            # if we have this value we need to store in next_iter_psi
            next_iter_psi[n_index] = psi_neighbour
            return
        calculate_w_updated_atc_mse(n_index, psi_neighbour)
        
        psi_bool = psi_bool | (1 << n_index) #that index has its psi published
    if STRATEGY == "CONSENSUS" or STRATEGY == "DIFFUSION_CTA":
        w_prev_neighbour = neighbour_payload["w_prev"]
        neighbour_ind = int(topic.split("_")[-1])
        w_prev_neighbour = [float(i) for i in w_prev_neighbour]
        n_index = n_map[neighbour_ind]
        has_got = (psi_bool >> n_index)&1
        if has_got or STATE=="HALT":
            next_iter_w_prev[n_index] = w_prev_neighbour
            return
        calculate_psi_con_mse(n_index, w_prev_neighbour)
        psi_bool = psi_bool | (1 << n_index) #that index has its psi published
    
    return

client = connectMQTT(callback_function)
#<---------TOPIC INITIALISATION------------------------->
def get_topic_name(node):
    topic_name = "publish_node_2_{}".format(node)
    return topic_name

client.publish(get_topic_name(node_index), msg="", retain=True, qos=0) #publish to self 
client.publish(SYNC_TOPIC, msg="", retain=True, qos = 0)
client.subscribe(SYNC_TOPIC)
#receives data via this topic
#<---------DATA PUBLISH------------------------->
def publish_psi_atc_mse(_iter = 0):
    ITER_PAYLOAD = {
        "p_id": p_id,
        "psi_iter": psi_iter
    }
    json_data = ujson.dumps(ITER_PAYLOAD)
    client.publish(get_topic_name(node_index), json_data, retain=True, qos=0)
    print("Publish done for ATC MSE PSI")

def publish_w_prev_con_cta_mse(_iter = 0):
    ITER_PAYLOAD = {
        "p_id": p_id,
        "w_prev": w_iter
    }
    json_data = ujson.dumps(ITER_PAYLOAD)
    client.publish(get_topic_name(node_index), json_data, retain=True, qos=0)
    if STRATEGY == "CONSENSUS":
        print("Publish done for CON MSE W_PREV")
    else:
        print("Publish done for CTA MSE W_PREV")
        
#<---------SENSOR CODE------------------------->
def get_low_cost_temp():
    dt.measure()
    return float(dt.temperature())

def get_high_cost_temp():
    high_cost_temp, pressure, humidity = bme.values
    return float(high_cost_temp)

#<---------DRIVER CODE------------------------->
#Initialiser 2: collects the last 5 measurements


def trainingDriver():
    global STRATEGY, CLEAR, psi_bool, STATE, neighbour_index, edge_weight, mu, MAX_ITERATIONS, w_prev, w_iter, psi_iter, psi_bool, n_map, N, M, p_id, client, next_iter_psi, next_iter_w_prev, NORM_FACTOR
    if STATE == "HALT":
        return
    time.sleep(1)
    for node in neighbour_index:
        client.publish(get_topic_name(node), msg="", retain=True, qos=0)
    #Subscribe to all neighboring topics
    for node in neighbour_index:
        client.subscribe(get_topic_name(node), qos=0)
    CREATION_PAYLOAD = {
        "device_id": node_id,
        "message_type": "registration",
        "p_id": p_id
    }
    CREATION_PAYLOAD = ujson.dumps(CREATION_PAYLOAD)
    client.publish(CENTRAL_TOPIC, msg=CREATION_PAYLOAD, retain=True, qos=0)
    time.sleep(2)
    u_iter = [0]*N
    for _iter in range(N):
        u_iter[_iter] = get_low_cost_temp() / NORM_FACTOR
        time.sleep(2)
    for _iter in range(MAX_ITERATIONS):
        if CLEAR == 1:
            client.publish(get_topic_name(node_index), "", retain = True, qos=0)
            CLEAR = 0
        if STATE == "HALT":
            return
        time.sleep(1)
        """
        this is one iteration
        0. Get a new measurement
        1. Compute the PSI for each iteration
        2. Publish the PSI to the respective topic
        3. Update the weight for the previous value
        4. Wait until all the other have published PSI
        """
        print("Iteration: {} *************************".format(_iter))
        psi_bool = 0
        u_new = get_low_cost_temp() / NORM_FACTOR
        u_iter.pop(0) #get rid of first index
        u_iter.append(u_new)
#         d = get_high_cost_temp()
        d = get_high_cost_temp() / NORM_FACTOR
        print("High Cost Data: {}".format(d))
        print("Low Cost Data: {}".format(u_iter))
        print(STRATEGY)
        if STRATEGY == "DIFFUSION_ATC":
            calculate_psi_atc_mse(d, u_iter, mu)
            print("PSI: {}".format(psi_iter))
            print("p_id: {}".format(p_id))
            publish_psi_atc_mse()
            
            w_prev = [w_i for w_i in w_iter]
            
            for _it in range(N):
                w_iter[_it] = edge_weight[M]*psi_iter[_it] #edge_weight[M] is contribution of self
            
            for _it in range(M):
                if len(next_iter_psi[_it]) != 0:
                    calculate_w_updated_atc_mse(_it, next_iter_psi[_it])
                    psi_bool = psi_bool | (1 << _it)
                    next_iter_psi[_it] = []
            while psi_bool != ((1<<M)-1):
                client.wait_msg()
                if CLEAR == 1:
                    client.publish(get_topic_name(node_index), "", retain = True, qos=0)
                    CLEAR = 0
                if STATE == "HALT":
                    return
                time.sleep(1)
        if STRATEGY == "CONSENSUS" or STRATEGY == "DIFFUSION_CTA":
            if _iter!=0:
                for _it in range(N):
                    psi_iter[_it] = edge_weight[M]*w_prev[_it]
                for _it in range(M):
                    if len(next_iter_w_prev[_it]) != 0:
                        calculate_w_updated_con_cta_mse(_it, next_iter_w_prev[_it], mu)
                        psi_bool = psi_bool | (1 << _it)
                        next_iter_w_prev[_it] = []
                while psi_bool != ((1<<M)-1):
                    client.wait_msg()
                    if STATE == "HALT":
                        return
                    time.sleep(1)
            calculate_w_updated_con_cta_mse(d, u_iter, mu)
            print("W_ITER: {}".format(w_iter))
            if _iter!=MAX_ITERATIONS-1:
                publish_w_prev_con_cta_mse()
                
        print("W_ITER: {}".format(w_iter))
        
        UPDATE_PAYLOAD = {
            "device_id": node_id,
            "message_type": "update",
            "p_id": p_id,
            "high_cost_data": d,
            "low_cost_data": u_iter,
            "w_iter": w_iter,
            "iteration": _iter+1
        }
        UPDATE_PAYLOAD = ujson.dumps(UPDATE_PAYLOAD)
        client.publish(CENTRAL_TOPIC, msg=UPDATE_PAYLOAD, retain=True, qos=0)
        time.sleep(1)
    print("Training Completed!")
    COMPLETION_PAYLOAD = {
        "device_id": node_id,
        "message_type": "completed",
        "p_id": p_id
    }
    COMPLETION_PAYLOAD = ujson.dumps(COMPLETION_PAYLOAD)
    client.publish(CENTRAL_TOPIC, msg=COMPLETION_PAYLOAD, retain=True, qos=0)
    time.sleep(2)

print("waiting...")
while True:
    client.wait_msg()
    if CLEAR == 1:
        client.publish(get_topic_name(node_index), "", retain = True, qos=0)
        CLEAR = 0
    if STATE == "TRAIN":
        print("training started!")
        trainingDriver()
        STATE = "HALT"
    
sensor_power_pin.value(0)





