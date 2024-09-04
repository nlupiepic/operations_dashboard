import streamlit as st
import utils
import datetime as dt
import pandas as pd
import streamlit_authenticator as stauth
import yaml

st.set_page_config(page_title="Epic Dashboard", layout='wide', page_icon='🛰') 

with open('config.yaml') as file:
    config = yaml.load(file, Loader=yaml.SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

name, authentication_status, username = authenticator.login('Login', 'main')

if authentication_status:
  authenticator.logout('Logout', 'sidebar')
  
  DATA_URL = 'https://drive.google.com/uc?export=download&id=10rbQufLi-1kvH3cIDZvWmouMiS3YYR4O'
  DATA_FILENAME = 'data.xlsx'
  TEAMS_URL = 'https://drive.google.com/uc?export=download&id=1EoHksGxR6zl9xMNsPfx7JbhDDqr36hyv'
  TEAMS_FILENAME = 'labor.xlsx'

  data, data_distr, months, quarters, cuentas = utils.load_data(url=DATA_URL, filename=DATA_FILENAME)
  data['fecha'] = data.fecha.apply(lambda fecha: dt.datetime.date(pd.to_datetime(fecha)))
  data_distr['fecha'] = data_distr.fecha.apply(lambda fecha: dt.datetime.date(pd.to_datetime(fecha)))
  months = [dt.datetime.date(month) for month in months]
  teams = utils.load_teams(url=TEAMS_URL, filename=TEAMS_FILENAME)

  refresh = st.sidebar.button('Actualizar Datos', help='Elimina Caché y actualiza datos desde Drive')
  if refresh:
    st.cache_data.clear()
    data, data_distr, months, quarters, cuentas = utils.load_data(url=DATA_URL, filename=DATA_FILENAME)
    data['fecha'] = data.fecha.apply(lambda fecha: dt.datetime.date(pd.to_datetime(fecha)))
    data_distr['fecha'] = data_distr.fecha.apply(lambda fecha: dt.datetime.date(pd.to_datetime(fecha)))
    months = [dt.datetime.date(month) for month in months]
    teams = utils.load_teams(url=TEAMS_URL, filename=TEAMS_FILENAME)

  st.sidebar.header("Navegación")

  pagina = st.sidebar.radio('', options=['Caja','Gastos','Aportes','Equipo','Configuración'], index=0)
  st.sidebar.subheader("Parámetros generales")

  moneda = st.sidebar.selectbox(
    'Moneda',
    ['USD','ARS']
  )

  sites = list(set(data.site))
  sites = st.sidebar.multiselect(
    'Empresa',
    sites,
    sites
  )

  devengado = st.sidebar.checkbox(label='Devengado', help='Distribuye gastos en el tiempo cuando corresponde. Por ejemplo, el caso de un depósito por un alquiler. No son números reales.')

  if devengado:
    data_aux, flow, stock = utils.filter(data_distr, sites, moneda.lower())
  else:
    data_aux, flow, stock = utils.filter(data, sites, moneda.lower())
    
  if pagina == 'Caja':
    st.title("Resumen de Cuentas")
    st.subheader('Estado de Caja')
    utils.caja(data_aux, flow, stock, moneda.lower())
  elif pagina == 'Gastos':
    st.title('Reporte de Gastos')
    date_range = st.sidebar.date_input('Rango de fechas', min_value=months[0], value=(dt.date(year=2021, month=1, day=1), data_aux.fecha.iloc[-1]), max_value=data_aux.fecha.values[-1])
    utils.gastos(data_aux, flow, moneda.lower(), date_range)
  elif pagina == 'Aportes':
    st.title('Aportes')
    date_range = st.sidebar.date_input('Rango de fechas', min_value=months[0], value=(months[0], data_aux.fecha.iloc[-1]), max_value=data_aux.fecha.values[-1])
    utils.aportes(data_aux, moneda.lower(), date_range)
  elif pagina == 'Equipo':
    st.title('Equipo')
    date_range = st.sidebar.date_input('Rango de fechas', min_value=months[0], value=(months[0], data_aux.fecha.iloc[-1]), max_value=data_aux.fecha.values[-1])
    utils.equipo(data_aux, teams, moneda.lower(), date_range)
  elif pagina == 'Configuración':
    st.title('Configuración')
    opcion = st.radio('', options=['','Cambiar contraseña'])
    if opcion == 'Cambiar contraseña':
      try:
        if authenticator.reset_password(username, 'Reset password'):
          st.success('Password modified successfully')
          with open('config.yaml', 'w') as file:
            yaml.dump(config, file, default_flow_style=False)
      except Exception as e:
        st.error(e)

elif authentication_status == False:
  st.error('Username/password is incorrect')
elif authentication_status == None:
  st.error('Please enter your username and password')