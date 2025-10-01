from spectrum_server.schema.radios import mongoOutput, tuneFrequencyOutput, timeDict, timeWaitOutput
import datetime
import pwd
import logging
import os
import time
import pymongo
from pymongo import MongoClient
from bson.json_util import dumps
from datetime import datetime, timezone
import socket
import struct
import os
import json  # added for JSON handling
import select


class Radio:

    def __init__(self):
        # Raise a clear error if the variable is missing
        try:
            self.hostname = os.environ["HOST_NAME"]
        except KeyError as e:
            raise EnvironmentError(
                "HOST_NAME environment variable not set"
            ) from e

    def tune_freq(self, center_freq_hz: int) -> tuneFrequencyOutput:
        """
        Send a retune message to the radio daemon.
        => --sample-rate-mhz [25, 50]
        => --atten-db [-1 => AUTO] OR [0,5,10,15,20,25,30]

        The original C++ implementation (tye_sp_ad_retune.cpp) builds a JSON
        payload with the following fields:
            - msg_type: "retune"
            - sample_rate_hz
            - center_freq_hz
            - atten_db
            - ref_level

        This Python version mirrors that behaviour.  Since the original
        Python signature only provides ``center_freq_hz``, the remaining
        fields are set to sensible defaults (0 for integers and 0.0 for the
        floating‑point reference level).  If callers need to customise those
        values they can extend this method or pass them via environment
        variables in the future.

        The function returns a ``tuneFrequencyOutput`` indicating success
        or failure based on whether the entire JSON payload was sent.
        """
        try:
        
        
            # Receive advertisement
            dst_ip, dst_port = self.recv_ad(61111)
            dst_port = 63333
            sample_rate_hz = 50000000
            atten_db = -1 #auto
            ref_level = -20
            
            # Send retune request and wait for status
            ok = self.send_retune_msg_wait_status(
                dst_ip,
                dst_port,
                sample_rate_hz,
                center_freq_hz,
                atten_db,
                ref_level,
            )
            if not ok:
                return tuneFrequencyOutput(success=False)

            return tuneFrequencyOutput(success=True)        

        except OSError:
            return tuneFrequencyOutput(success=False)
        finally:
            # Ensure the socket is closed even on failure
            try:
                sock.close()
            except Exception:
                pass




    def get_time(self) -> timeDict:
        ns = time.time_ns()
        seconds = ns / 1_000_000_000
        dt_obj = datetime.fromtimestamp(seconds, timezone.utc)
        out = timeDict(nanoseconds=ns,utc_datetime=dt_obj)
        return out
            

    def wait_time(self, seconds_to_wait: int) -> timeWaitOutput:            
        time.sleep(seconds_to_wait)
        return timeWaitOutput(success=True)


    def get_results(self, center_freq_hz: int) -> mongoOutput:
        host       = self.hostname
        port       = 27018     
        target_db  = "tye_sp"  

        client = MongoClient(host,port)
    
        db = client[target_db]
        collections = db.list_collection_names()
        collections = sorted(collections)
        collection = db[collections[-1]]
    
        pipeline = [

            {
              "$match": {
              "radio_center_freq_hz": center_freq_hz
                }
            },
            {
              "$match": {
              "spectrogram_diff":{'$exists':True}
              }
            },
            {
               "$sort": {
               "start_time_ns": pymongo.ASCENDING
               }
            },
            {
                "$limit": 10
            }     
            ]
        
        db_cursor = collection.aggregate(pipeline)
        output = mongoOutput(mongoResults=dumps(db_cursor))
        return output
        
    def recv_ad(self, ad_port):
        """
        Listen for an advertisement JSON on the given UDP port.
        Returns (dst_ipaddr, dst_port) on success, or (None, None) on failure/timeout.
        """
        print(">> OPENING AD SOCKET ", end="", flush=True)

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        except OSError as e:
            print("[FAIL]")
            print(f"Socket creation error: {e}")
            return None, None

        # Set socket options
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setblocking(False)

        try:
            ip = socket.gethostbyname(self.hostname)
            sock.bind((str(ip), ad_port))
        except OSError as e:
            print("[FAIL]")
            print(f"Bind error on port {ad_port}: {e}")
            sock.close()
            return None, None

        print("[OK]")
        print(">> WAITING FOR AD ", end="", flush=True)

        deadline = time.time() + 2
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                print("\n[TIMEOUT]")
                sock.close()
                return None, None

            rlist, _, _ = select.select([sock], [], [], remaining)
            if not rlist:
                continue

            try:
                data, addr = sock.recvfrom(1024)
            except OSError:
                continue

            try:
                msg = json.loads(data.decode())
            except json.JSONDecodeError:
                print("\n[PARSE ERROR] [FAIL]")
                continue

            # Validate message structure
            if not isinstance(msg, dict):
                print("\n[MSG IS NOT AN OBJECT] [FAIL]")
                continue

            if msg.get("msg_type") != "ad":
                # Not an advertisement; ignore
                continue

            if "retune_port" not in msg:
                print("\n[KEY \"retune_port\" NOT FOUND] [FAIL]")
                continue

            if not isinstance(msg["retune_port"], int):
                print("\n[KEY \"retune_port\" INCORRECT TYPE] [FAIL]")
                continue

            print(f"\n[OK] => {data.decode()}")
            dst_ipaddr = socket.inet_aton(addr[0])  # bytes representation
            dst_ip = struct.unpack("!I", dst_ipaddr)[0]  # uint32 network order
            dst_port = int(msg["retune_port"])
            sock.close()
            return dst_ip, dst_port
            
    def send_retune_msg_wait_status(self, dst_ip, dst_port, sample_rate_hz,
                                   center_freq_hz, atten_db, ref_level):
        """
        Send a retune JSON message to the destination and wait for a status reply.
        Returns True on success, False otherwise.
        """
        print(">> OPENING RETUNE SOCKET ", end="", flush=True)

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        except OSError as e:
            print("[FAIL]")
            print(f"Socket creation error: {e}")
            return False

        sock.setblocking(False)
        print("[OK]")

        # Build retune JSON
        print(">> BUILDING RETUNE MSG ", end="", flush=True)
        msg = {
            "msg_type": "retune",
            "sample_rate_hz": sample_rate_hz,
            "center_freq_hz": center_freq_hz,
            "atten_db": atten_db,
            "ref_level": ref_level,
        }

        # Ensure decimal places similar to C++ Writer setting
        json_msg = json.dumps(msg, separators=(',', ':'), ensure_ascii=False)
        print("[OK]")

        # Send the message
        print(">> SENDING RETUNE MSG ", end="", flush=True)
        #dst_ip_str = socket.inet_ntoa(struct.pack("!I", dst_ip))
        try:
            dst_ip_str = str(socket.gethostbyname(self.hostname))
            sent = sock.sendto(json_msg.encode(), (dst_ip_str, dst_port))
        except OSError as e:
            print("[FAIL]")
            print(f"Send error: {e}")
            sock.close()
            return False

        if sent != len(json_msg):
            print("[FAIL]")
            sock.close()
            return False

        print(f"[OK] DST [{dst_ip_str}:{dst_port}] MSG {json_msg}")

        # Wait for status reply
        print(">> WAITING FOR RETUNE STATUS ", end="", flush=True)
        deadline = time.time() + 2
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                print("\n[TIMEOUT]")
                sock.close()
                return False

            rlist, _, _ = select.select([sock], [], [], remaining)
            if not rlist:
                continue

            try:
                data, _ = sock.recvfrom(256)
            except OSError:
                continue

            try:
                status_msg = json.loads(data.decode())
            except json.JSONDecodeError:
                print("\n[PARSE ERROR] [FAIL]")
                break

            if not isinstance(status_msg, dict):
                print("\n[MSG IS NOT AN OBJECT] [FAIL]")
                break

            if "msg_type" not in status_msg or "status" not in status_msg:
                missing_key = "msg_type" if "msg_type" not in status_msg else "status"
                print(f"\n[KEY \"{missing_key}\" NOT FOUND] [FAIL]")
                break

            msg_type = status_msg["msg_type"]
            status = status_msg["status"]

            if msg_type != "retune_status":
                print("\n[MSG TYPE INCORRECT] [FAIL]")
                break

            if status == "success":
                print("\n[SUCCESS]")
            else:
                print("\n[FAIL]")
            break

        sock.close()
        return True                    
