#---------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------

import sys
import time
import signal
import socket
import json

from pymongo import MongoClient

#-- variables --------------------------------------------------------------------------------------------------------------

g_ad_socket          = None
g_cnc_port           = 0
g_retune_port        = 0
g_db_connect_uri     = ""
g_db_name            = ""
g_db_collection_name = ""
g_db_client          = None
g_db                 = None
g_db_collection      = None
g_exit_requested     = False

#-- support functions ------------------------------------------------------------------------------------------------------

# handle control-c
def exit_handler( signal: int, stack_frame ) -> None:

    global g_exit_requested
    g_exit_requested = True

#---------------------------------------------------------------------------------------------------------------------------

# display usage
def display_usage() -> None:

    print("", flush=True)
    print("USAGE: python3 test_wait_ad_connect_db.py [ad-port]", flush=True)
    print("", flush=True)
    print("  * ad-port [OPTIONAL] Port on which to receive advertisements (Default is 61111)", flush=True)
    print("", flush=True)

#---------------------------------------------------------------------------------------------------------------------------

# open socket on which to receive advertisements
def open_ad_socket( ad_port: int ) -> bool:

    global g_ad_socket

    ok = True

    try:

        # open and bind UDP socket
        g_ad_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        g_ad_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        g_ad_socket.bind(("0.0.0.0", ad_port))

        print(">> WAITING FOR AD [PORT " + str(ad_port) + "]", flush=True)

    except Exception as e:

        print(">> !! EXCEPTION !! " + str(e), flush=True)
        ok = False

    return ( ok )

#---------------------------------------------------------------------------------------------------------------------------

# close advertisement socket
def close_ad_socket() -> None:

    global g_ad_socket

    if ( g_ad_socket != None ): g_ad_socket.close()

#---------------------------------------------------------------------------------------------------------------------------

# disconnect from the database
def disconnect_db() -> None:

    global g_db_client

    if ( g_db_client != None ): g_db_client.close()

#---------------------------------------------------------------------------------------------------------------------------

# handle the WAIT-AD state
def handle_state_wait_ad() -> bool:

    global g_ad_socket
    global g_cnc_port
    global g_retune_port
    global g_db_connect_uri
    global g_db_name
    global g_db_collection_name

    ok = True

    try:

        # attempt to receive an advertisement message
        g_ad_socket.settimeout(0.2)
        msg, from_addr = g_ad_socket.recvfrom(1024)

        # decode and load the message as JSON
        json_msg = json.loads(msg.decode('utf-8'))

        # extract data fields from the JSON message
        g_cnc_port           = json_msg['cnc_port']
        g_retune_port        = json_msg['retune_port']
        g_db_connect_uri     = json_msg['db_connect_uri']
        g_db_name            = json_msg['db_name']
        g_db_collection_name = json_msg['db_collection']

        # display JSON data fields
        print("   [AD] CNC PORT [" + str(g_cnc_port) + "]", flush=True)
        print("   [AD] RETUNE PORT [" + str(g_retune_port) + "]", flush=True)
        print("   [AD] DB CONNECT URI [" + g_db_connect_uri + "]", flush=True)
        print("   [AD] DB NAME [" + g_db_name + "]", flush=True)
        print("   [AD] DB COLLECTION [" + g_db_collection_name + "]", flush=True)
        print("", flush=True)

    except Exception as e: ok = False # most likely a receive timeout

    return ( ok )

#---------------------------------------------------------------------------------------------------------------------------

# handle the CONNECT-DB state
def handle_state_connect_db() -> bool:

    global g_db_connect_uri
    global g_db_name
    global g_db_collection_name
    global g_db_client
    global g_db
    global g_db_collection

    ok = True

    try:

        print(">> CONNECTING TO DB [" + g_db_connect_uri + "/" + g_db_name + "/" + g_db_collection_name + "] ",
              end='', flush=True)

        g_db_client     = MongoClient(g_db_connect_uri)
        g_db            = g_db_client[g_db_name]
        g_db_collection = g_db[g_db_collection_name]

        print("[OK]", flush=True)

    except Exception as e:

        print("[FAIL] !! EXCEPTION !! " + str(e), flush=True)
        ok = False

    return ( ok )

#---------------------------------------------------------------------------------------------------------------------------

# handle the QUERY-DB state
def handle_state_query_db() -> bool:

    global g_db_collection

    ok = True

    try:

        print(">> QUERYING DB", end='', flush=True)
        query_result = g_db_collection.find_one()

        print("", flush=True)
        print("", flush=True)
        print(query_result, flush=True)

    except Exception as e:

        print("[FAIL] !! EXCEPTION !! " + str(e), flush=True)
        ok = False

    return ( ok )

#---------------------------------------------------------------------------------------------------------------------------

# handle the LOOP state
def handle_state_loop() -> bool:

    return ( True )

#-- entry point ------------------------------------------------------------------------------------------------------------

def main( ad_port: int ) -> None:

    global g_exit_requested

    # set state IDs
    STATE_WAIT_AD    = int(1)
    STATE_CONNECT_DB = int(2)
    STATE_QUERY_DB   = int(3)
    STATE_LOOP       = int(4)

    # set the current state
    state = STATE_WAIT_AD
    print()

    # open socket to receive advertisements
    ok = open_ad_socket(ad_port)
    if ( ok ):

        while ( True ): # forever

            # bail if exit was requested
            if ( g_exit_requested ): break

            # handle things based on the current state
            if ( state == STATE_WAIT_AD ):

                ok = handle_state_wait_ad()
                if ( ok ): state = STATE_CONNECT_DB

            elif ( state == STATE_CONNECT_DB ):

                ok = handle_state_connect_db()
                if ( ok ): state = STATE_QUERY_DB

            elif ( state == STATE_QUERY_DB ):

                time.sleep(3.0)

                ok = handle_state_query_db()
                if ( ok ): state = STATE_LOOP

            elif ( state == STATE_LOOP ):

                handle_state_loop()

            time.sleep(0.1)

        # shutdown
        print("", flush=True)
        print(">> SHUTTING DOWN", flush=True)

        close_ad_socket()
        if ( state == STATE_LOOP ): disconnect_db()

        print("", flush=True)

#---------------------------------------------------------------------------------------------------------------------------

if ( __name__ == '__main__' ):

    # set the exit/control-c handler
    signal.signal(signal.SIGINT, exit_handler)

    # handle command line arguments
    num_cmdline_args = len(sys.argv)
    if ( (num_cmdline_args == 1) or (num_cmdline_args == 2) ):

        if ( num_cmdline_args == 2 ): ad_port = int(sys.argv[1])
        else:                         ad_port = int(61111)

        main(ad_port)

    else: display_usage()

#---------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------
