import pandas as pd
from sqlalchemy import create_engine
#from streamlit.report_thread import get_report_ctx
import streamlit as st
from streamlit.scriptrunner.script_run_context import get_script_run_ctx, add_script_run_ctx
import os
import psycopg2

def get_session_id():
    session_id = get_script_run_ctx().session_id
    #session_id = get_report_ctx().session_id
    session_id = session_id.replace('-','_')
    session_id = '_id_' + session_id
    return session_id


#get_script_run_ctx().session_id

def write_state(column,value,engine,data_base_name):
    engine.execute("UPDATE %s SET %s='%s'" % (data_base_name,column,value))

def write_state_df(df,engine,data_base_name):
    df.to_sql('%s' % (data_base_name),engine,index=False,if_exists='append',chunksize=1000)

def read_state(column,engine,data_base_name):
    state_var = engine.execute("SELECT %s FROM %s" % (column,data_base_name))
    state_var = state_var.first()[0]
    return state_var

def read_state_df(engine,data_base_name):
    try:
        df = pd.read_sql_table(data_base_name,con=engine)
    except:
        df = pd.DataFrame([])
    return df