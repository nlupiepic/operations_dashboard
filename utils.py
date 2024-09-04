import os
import wget
import streamlit as st
import streamlit.components.v1 as components
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
import pandas as pd
import numpy as np
import datetime as dt
from dateutil.relativedelta import relativedelta
import re
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from pivottablejs import pivot_ui
import streamlit_authenticator as stauth
import yaml

colores = ['#1f77b4', '#ff7f0e', '#2ca02c','#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
META = dict()

def normalize(s):
    replacements = (
        ('á','a'),
        ('é','e'),
        ('í','i'),
        ('ó','o'),
        ('ú','u')
    )

    for a, b in replacements:
        s = s.replace(a, b).replace(a.upper(), b.upper())
    return s

def process_string(s):
    if pd.isna(s):
        return s
    else:
        return normalize( re.sub(' +', ' ', str(s).lower()) )

def distribute_over_months(data):
    over_months = data[~pd.isna(data.over_months)]
    rows = []
    for idx, row in over_months.iterrows():
        months = row.over_months
        monto = row.usd / months
        row.over_months = np.nan
        row.usd = monto
        data.loc[idx] = row
        for _ in np.arange(1, months):
            row.month = row.month + relativedelta(months=+1)
            row.fecha = row.fecha + relativedelta(months=+1)
            rows.append( row.copy() )
            
    data = pd.concat([data, pd.DataFrame.from_records(rows)], ignore_index=True)
    data.reset_index(drop=True, inplace=True) 
    data.drop(data[data.fecha >= dt.datetime.today()].index, inplace=True)
    
    return data.sort_values(by=['fecha','id']).reset_index(drop=True)

class Proyecto():
  def __init__(self, data, months):
    
    self.sub_proyectos = set(data.sub_proyecto_1)
    self.flow = pd.DataFrame(index=months)
    flow_aux = pd.DataFrame(index=pd.unique(data.month))
    sumas = pd.pivot_table( data, values='usd', index='month', columns=['sub_proyecto_1','destino'], aggfunc=sum, fill_value=0 )
    restas = pd.pivot_table( data, values='usd', index='month', columns=['sub_proyecto_1','origen'], aggfunc=sum, fill_value=0 )
    restas.columns.names = ['sub_proyecto_1','destino']
    tmp = sumas.sub(restas, axis=1, fill_value=0)
    flow_aux[list(self.sub_proyectos)]  = np.array([tmp[sub_proyecto][list(set(tmp[sub_proyecto].columns)&META['cuentas_gastos'])].sum(axis=1) for sub_proyecto in self.sub_proyectos]).transpose()
    self.flow = self.flow.join(flow_aux).fillna(0)
    self.stock = self.flow.cumsum()

class Proyectos():
  def __init__(self, data):
    self.flow = pd.DataFrame(index=pd.unique(data.month))
    self.names = set(data.proyecto)
    sumas = pd.pivot_table( data, values='usd', index='month', columns=['proyecto','destino'], aggfunc=sum, fill_value=0)
    restas = pd.pivot_table( data, values='usd', index='month', columns=['proyecto','origen'], aggfunc=sum, fill_value=0)
    restas.columns.names = ['proyecto','destino']
    tmp = sumas.sub(restas, axis=1, fill_value=0)
    
    self.flow[list(self.names)] = np.array([tmp[proyecto][list(set(tmp[proyecto].columns)&META['cuentas_gastos'])].sum(axis=1) for proyecto in self.names]).transpose()
    self.flow['Outflows'] = self.flow.sum(axis=1)
    self.stock = self.flow.cumsum()
    self.flow['MA'] = self.flow['Outflows'].rolling(window=3).mean()
    #self.flow.iloc[-1,-1] = self.flow.iloc[-2,-1]

    self.proyectos = {proyecto: Proyecto(data[data.proyecto==proyecto], pd.unique(data.month)) for proyecto in self.names } 

@st.cache_data
def load_data(url, filename):
    global META
    
    if os.path.exists(filename):
        os.remove(filename)
    wget.download(url, filename)

    META = pd.read_excel('data.xlsx', sheet_name='meta', header=None, index_col=0)
    META = META[1].to_dict()

    for key, value in META.items():
        if key not in ['sheet_names','site_names']:
            META[key] = set(value.split(','))
        else:
            META[key] = value.split(',')

    META['cuentas_gastos'] = META['OPEX'].union(META['Mission_costs'], META['Otros_gastos'], META['FOPEX'], META['CAPEX'], META['Hardware'])
    META['activo'] = META['Caja'].union(META['Mission_costs'], META['OPEX'], META['Otros_gastos'], META['FOPEX'], META['CAPEX'], META['Hardware'], META['Transferencias'])
    META['pasivo'] = META['Aportes'].union(META['Deudas'], META['Otros_ingresos'])

    datasets = [pd.read_excel('data.xlsx', sheet_name=sheet_name, header=2) for sheet_name in META['sheet_names']]
    for dataset, site_name in zip(datasets, META['site_names']):
        dataset['site'] = site_name
        
    data = pd.concat(datasets, ignore_index=True)
    
    data.columns = [col.lower().replace(' ', '_') for col in data.columns]
    data.drop(['year','month'], axis=1, inplace=True)
    data = data.sort_values('fecha').reset_index(drop=True)
    data['month'] = data.fecha.apply(lambda fecha: fecha.replace(day=1))
    data['quarter'] = data.month.apply(lambda fecha: fecha.replace(month=int(np.ceil(fecha.month/3))))
    data['cuenta'] = data.cuenta.apply(process_string)
    data['detalle'] = data.detalle.apply(process_string)
    data['proveedor'] = data.proveedor.apply(process_string)
    data['destino'] = data.destino.apply(process_string)
    data['origen'] = data.origen.apply(process_string)
    months = pd.to_datetime(pd.unique(data.month))
    quarters = pd.unique(data.quarter)
    cuentas = set(data.destino).union(set(data.origen))

    data_distr = distribute_over_months(data.copy())

    return data, data_distr, months, quarters, cuentas

@st.cache_data
def load_teams(url, filename):
    if os.path.exists(filename):
        os.remove(filename)
    wget.download(url, filename)

    labor = pd.read_excel(filename, sheet_name='Roles', header=0, index_col=0)
    labor = labor.transpose()
    labor.drop(labels=np.nan, axis=1, inplace=True)

    teams = pd.read_excel(filename, sheet_name='Teams', header=0, index_col=0)
    teams = teams.transpose()
    teams.drop(labels=np.nan, axis=1, inplace=True)

    # L = pd.DataFrame(labor.sum(axis=1), columns=['L'])
    # tech = pd.DataFrame(
    #     labor[labor.columns[~labor.columns.isin(['Management', 'Accounting', 'Media & Sales'])]].sum(axis=1),
    #     columns=['tech']
    # )
    # sga = pd.DataFrame(
    #     labor[labor.columns[labor.columns.isin(['Management', 'Accounting', 'Media & Sales'])]].sum(axis=1),
    #     columns=['sga']
    # )

    # prop_labor = pd.DataFrame(
    #     labor[['Mech Eng', 'Mech Tech', 'Other Eng', 'Other Tech']].sum(axis=1),
    #     columns=['prop']
    # )

    # elec_labor = pd.DataFrame(
    #     labor[['Elec Eng', 'Elec Tech']].sum(axis=1),
    #     columns=['elec']
    # )

    return teams

@st.cache_data
def filter(data, sites, moneda):
    data = data[data.site.isin(sites)].reset_index(drop=True)
    flow = pd.pivot_table( data, values=moneda, index='month', columns='destino', aggfunc=sum, fill_value=0).sub( \
           pd.pivot_table( data, values=moneda, index='month', columns='origen', aggfunc=sum, fill_value=0), \
           axis=1, fill_value=0)

    flow['Caja'] = flow[list(set(flow.columns)&META['Caja'])].sum(axis=1)
    flow['Mission_costs'] = flow[list(set(flow.columns)&META['Mission_costs'])].sum(axis=1)
    flow['FOPEX'] = flow[list(set(flow.columns)&META['FOPEX'])].sum(axis=1)
    flow['OPEX'] = flow[list(set(flow.columns)&META['OPEX'])].sum(axis=1)
    flow['CAPEX'] = flow[list(set(flow.columns)&META['CAPEX'])].sum(axis=1)
    flow['Hardware'] = flow[list(set(flow.columns)&META['Hardware'])].sum(axis=1)
    flow['Otros_gastos'] = flow[list(set(flow.columns)&META['Otros_gastos'])].sum(axis=1)

    flow['Outflows'] = flow[['Mission_costs','FOPEX','OPEX','CAPEX','Hardware','Otros_gastos']].sum(axis=1)
    
    stock = flow.cumsum()
    
    flow['MA'] = flow['Outflows'].rolling(window=3).mean()
    #flow.iloc[-1,-1] = flow.iloc[-2,-1]

    return data.copy(), flow, stock

@st.cache_data
def get_proyectos(data):
    proyectos = Proyectos(data)
    return proyectos

def caja(data, flow, stock, moneda):
    cuenta = st.selectbox(label='Cuenta', options=list(map(str.title, ['Todas']+list(META['Caja']))), index=0).lower()
    
    if cuenta == 'todas':
        cuenta = 'Caja'

    st.write('Saldo actual:', moneda.upper(), '{:,.2f}'.format(stock.tail(1)[cuenta].values[0]))

    ## Estado de Caja
    
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=stock.index,
            y=stock[cuenta],
            name='Stock'
        )
    )

    fig.add_trace(
        go.Bar(
            x=flow.index,
            y=flow[cuenta],
            name='Flujo'
        )
    )

    fig.update_yaxes(title_text=moneda.upper())
    fig.update_layout(title='<b>Estado de {}</b>'.format(cuenta.title()))
    st.plotly_chart(fig, use_container_width=True)


    ## Burn & Runway
    
    st.subheader('Burn & Runway')
    cost_names = ['Todos','Mission_costs','FOPEX','OPEX','CAPEX','Hardware','Otros_gastos']
    cuentas_elegidas = st.multiselect('Cuentas para Burn', cost_names, ['FOPEX'])
    
    aux_flow = flow.copy()
    
    if 'Todos' in cuentas_elegidas:
        cuentas_elegidas = cost_names
    else:
        aux_flow.Outflows = aux_flow[cuentas_elegidas].sum(axis=1)
        aux_flow.MA = aux_flow.Outflows.rolling(window=3).mean()
        aux_flow.iloc[-1,-1] = aux_flow.iloc[-2,-1]

    st.write('Burn actual:', moneda.upper(), '{:,.0f} por mes'.format(aux_flow.tail(1)['MA'].values[0]))
    st.write('Runway: {:,.0f} meses'.format((stock.tail(1)['Caja'] / aux_flow.tail(1)['MA']).values[0]))

    fig = make_subplots( specs = [[{'secondary_y':True}]] )

    fig.add_trace(
        go.Scatter(
            x=aux_flow.index,
            y=aux_flow.MA,
            name='Burn'
        ),
        secondary_y=False
    )

    fig.add_trace(
        go.Scatter(
            x=aux_flow.index,
            y=stock.Caja / aux_flow.MA,
            name='Runway'
        ),
        secondary_y=True
    )

    fig.update_layout(title='<b>Monthly Burn - Trailing 3 Months MA</b>')
    fig.update_yaxes(title_text="US$", secondary_y=False)
    fig.update_yaxes(title_text="Months", secondary_y=True)

    st.plotly_chart(fig, use_container_width=True)


    ## Ultimos Movimientos

    st.subheader('Últimos movimientos de {}:'.format(cuenta.title()))

    flujo_nombre = 'flujo (' + moneda + ')'
    stock_nombre = 'stock (' + moneda + ')'
    
    if cuenta == 'Caja':
        mayor = data[(data.destino.isin(META['Caja'])) | (data.origen.isin(META['Caja']))].reset_index(drop=True).copy()
        mayor[flujo_nombre] = mayor[moneda] * ( (mayor.origen.isin(META['Caja']))*-1 + (mayor.destino.isin(META['Caja']))*1 )
        
    else:
        mayor = data[(data.destino == cuenta) | (data.origen == cuenta)].reset_index(drop=True).copy()
        mayor[flujo_nombre] = mayor[moneda] * ( (mayor.origen == cuenta)*-1 + (mayor.destino == cuenta)*1 )

    mayor.sort_values('fecha', ascending=True, inplace=True)
    mayor[stock_nombre] = mayor[flujo_nombre].cumsum()
    mayor = mayor[['id','fecha',flujo_nombre,stock_nombre,'categoria','sub_categoria_1','proyecto','cuenta','proveedor','detalle','comprobante','site']]
    #mayor[flujo_nombre] = mayor[flujo_nombre].map('${:,.2f}'.format)
    #mayor[stock_nombre] = mayor[stock_nombre].map('${:,.2f}'.format)
    mayor[flujo_nombre] = mayor[flujo_nombre].round(2)
    mayor[stock_nombre] = mayor[stock_nombre].round(2)
    mayor = mayor[::-1].reset_index(drop=True)
    mayor['id'] = mayor.index
    mayor.fillna('', inplace=True)
    
    gb = GridOptionsBuilder.from_dataframe(mayor)
    gb.configure_pagination()
    gb.configure_side_bar()
    gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc='sum', editable=True)
    gridOptions = gb.build()
    
    #gridOptions['columnDefs'] = [{'field':col, 'pivot':True, 'value':True} if col in ['categoria','sub_categoria_1'] else \
    #    {'field':col, 'pivot':False, 'value':True} for col in mayor.columns]
    AgGrid(mayor, gridOptions = gridOptions)#, enable_enterprise_modules=True)


def gastos(data, flow, moneda, date_range): 
    ## Por categoria
    
    fig = go.Figure(data=[
                        go.Bar(
                            name=cuenta,
                            x=flow.index,
                            y=flow[cuenta],
                            hoverinfo='text',
                            text=['{}<br>Total: ${:,.0f} <br>{}: ${:,.0f}'.format(date.strftime('%b-%Y'), total, cuenta, cat) for date, total, cat in zip(flow.index, flow.Outflows, flow[cuenta])])
                            for cuenta in ['Mission_costs','FOPEX','OPEX','Hardware','CAPEX','Otros_gastos']
                    ])
    
    fig.add_trace(
        go.Scatter(
            x=flow.index,
            y=flow.MA,
            hoverinfo='text',
            line_color='darkorange',
            text=['{}<br>MA: ${:,.0f}'.format(date.strftime('%b-%Y'), ma) for date, ma in zip(flow.index, flow.MA)],
            name='MA'
        )
    )

    fig.update_layout(barmode='stack', title='<b>Gastos Mensuales por Categoría</b>')#, height=500)
    fig.update_yaxes(title_text=moneda.upper())
    st.plotly_chart(fig, use_container_width=True)

    ## Por proyectos

    data = data[data.destino.isin(META['cuentas_gastos'])].reset_index(drop=True).copy()

    proyectos = get_proyectos(data)
    fig = go.Figure(data=[
                      go.Bar(
                          name=col,
                          x=proyectos.flow.index,
                          y=proyectos.flow[col],
                          hoverinfo='text',
                          text=['{}<br>Total: ${:,.0f} <br>{}: ${:,.0f}'.format(date.strftime('%b-%Y'), total, col, cat) for date, total, cat in zip(proyectos.flow.index, proyectos.flow.Outflows, proyectos.flow[col])]) for col in proyectos.names
                     ])
    fig.add_trace(
        go.Scatter(
            x=proyectos.flow.index,
            y=proyectos.flow.MA,
            hoverinfo='text',
            line_color='darkorange',
            text=['{}<br>MA: ${:,.0f}'.format(date.strftime('%b-%Y'), ma) for date, ma in zip(proyectos.flow.index, proyectos.flow.MA)],
            name='MA'
        )
    )

    fig.update_layout(barmode='stack')
    fig.update_layout(title='<b>Gastos Mensuales por Destino</b>')
    fig.update_yaxes(title_text=moneda.upper())
    st.plotly_chart(fig, use_container_width=True)

    ## Planned vs. Executed

    start, end = pd.to_datetime('2021-07-01'), data.month.iloc[-1]

    if set(pd.unique(data.site)) == set(META['site_names']): # Solo lo muestro si elijo todos los sitios

        salaries = flow.loc[start:end,flow.columns[flow.columns.str.contains('salaries')]].sum().sum()
        fopex_ex_salaries = flow.loc[start:end,'FOPEX'].sum().sum() - salaries
        opex = flow.loc[start:end,'OPEX'].sum().sum()
        tools = flow.loc[start:end,'herramientas'].sum().sum()
        machinery = flow.loc[start:end,['maquinaria','utilaje']].sum().sum()
        infraestructure = flow.loc[start:end,['infraestructura','mano de obra']].sum().sum()
        office_eq = flow.loc[start:end,['equipo de oficina', 'rodados']].sum().sum()
        test_eq = flow.loc[start:end,'test equipment'].sum().sum()
        vehicle_rd = flow.loc[start:end,'vehicle r&d'].sum().sum()
        general_rd = flow.loc[start:end,list(META['general_rd'])].sum().sum()
        prop_prod = flow.loc[start:end,'propellant production hardware'].sum().sum()
        flight_tugs = flow.loc[start:end,'flight tugs'].sum().sum()
        ext_flt = flow.loc[start:end,'ext. flt. hardware'].sum().sum()
        vehicle_dev = flow.loc[start:end,'vehicle development'].sum().sum()
        launch_costs = flow.loc[start:end,'rideshare costs'].sum().sum()
        mission_legal = flow.loc[start:end,'mission legal costs'].sum().sum()
        mission_ops = flow.loc[start:end,'mission ops'].sum().sum()

        planned_executed = pd.DataFrame(
                            index=['Planned','Executed'],
                            data={
                                'Salaries': [861173, salaries],
                                'FOPEX (Ex. Salaries)': [115275, fopex_ex_salaries],
                                'OPEX': [115558, opex],
                                'Tools': [36849, tools],
                                'Machinery': [60750, machinery],
                                'Infraestructure': [61050, infraestructure],
                                'Office Eq.': [20000, office_eq],
                                'Test Eq.': [94680, test_eq],
                                'Vehicle R&D': [51400, vehicle_rd],
                                'General R&D': [50000, general_rd],
                                'Prop. Prod. Hardware': [15000, prop_prod],
                                'Flight Tugs': [250000, flight_tugs],
                                'Ext. Flt. Hardware': [100000, ext_flt],
                                'Vehicle Dev': [110000, vehicle_dev],
                                'Launch Costs': [800000, launch_costs],
                                'Mission Legal Costs': [50000, mission_legal],
                                'Mission OPS': [70000, mission_ops]
                            }
                        ).transpose().round(-1)
        
        planned_executed = pd.concat([
                            planned_executed,
                            pd.DataFrame(planned_executed.sum(), columns=['Total:']).transpose()
                        ])

        planned_executed['diff'] = planned_executed.Executed - planned_executed.Planned
        st.title('Gasto Planeado Vs. Ejecutado desde Julio 2021')
        st.dataframe(planned_executed.astype(int), use_container_width=True)
    

    ## Tabla Resumen

    pivot = pivot_ui(
        data.loc[
            data.fecha.between(date_range[0], date_range[1]),
            ['categoria',
            'sub_categoria_1',
            'proyecto',
            'sub_proyecto_1',
            'sub_proyecto_2',
            'sub_proyecto_3',
            'sistema',
            'destino',
            'cuenta',
            'proveedor',
            'detalle',
            'usd',
            'month',
            'site'
        ]],
        rows=['proyecto','sub_proyecto_1','categoria'],
        #cols=['categoria'],
        vals=['usd'],
        aggregatorName='Sum',
        outfile_path='/tmp/pivottablejs.html'
        )
    st.title('Tabla Resumen')

    
    st.write('Para el período ' + str(date_range[0]) + ' - ' + str(date_range[1]))
    
    with open(pivot.src) as pivot:
        components.html(pivot.read(), height=1000, scrolling=True)

    st.title('Proyectos')
    st.write('Para el período ' + str(date_range[0]) + ' - ' + str(date_range[1]))
        
    

    #data['month'] = data.month.apply(lambda fecha: dt.datetime.date(pd.to_datetime(fecha)))
    data_proyectos = data[
                            (data.fecha.between(date_range[0], date_range[1]))
                            #(data.proyecto.isin(proyectos_elegidos))
                        ].sort_values(['fecha','id']).reset_index(drop=True).copy()

    proyectos = get_proyectos(data_proyectos)

    # subproyectos = st.checkbox(label='Visualizar Sub Proyectos')
    # if subproyectos:
    #     fig = go.Figure()
    #     i=0

    #     for proyecto in proyectos.names:
    #         for sub_proyecto in proyectos.proyectos[proyecto].sub_proyectos:
    #             fig.add_trace(
    #             go.Scatter(
    #                 x=proyectos.proyectos[proyecto].stock.index,
    #                 y=proyectos.proyectos[proyecto].stock[sub_proyecto],
    #                 name=sub_proyecto + ' ' + proyecto,
    #                 stackgroup='one',
    #                 line_color=colores[ i % len(colores) ]
    #             )
    #         )
    #         i+=1
    # else:
    #     fig = go.Figure(
    #         data = [go.Scatter(name=proyecto, x=proyectos.stock.index, y=proyectos.stock[proyecto], stackgroup='one') for proyecto in proyectos.names]        
    #     )
    
    fig = go.Figure(data=[
                      go.Bar(
                          name=col,
                          x=proyectos.flow.index,
                          y=proyectos.flow[col],
                          hoverinfo='text',
                          text=['{}<br>Total: ${:,.0f} <br>{}: ${:,.0f}'.format(date.strftime('%b-%Y'), total, col, cat) for date, total, cat in zip(proyectos.flow.index, proyectos.flow.Outflows, proyectos.flow[col])]) 
                          for col in proyectos.names
                     ])


    fig.update_layout(barmode='stack')
    
    fig.update_layout(title='<b>Evolución de Proyectos</b>')
    fig.update_yaxes(title_text=moneda.upper())
    st.plotly_chart(fig, use_container_width=True)

    ## Treemaps

    st.subheader('Treemaps')

    with st.form(key = 'Form'):
        #with st.expander('Proyectos'):
        proyectos = ['Todos'] + list(set(data.proyecto))
        
        proyectos_elegidos = st.multiselect('Proyectos Elegidos', proyectos, ['Todos'])
        if 'Todos' in proyectos_elegidos:
            proyectos_elegidos = proyectos
        
        campos1 = st.multiselect(
            'Campos Gráfico 1 (el orden importa)',
            ['Categoria','Sub_categoria_1','Proyecto','Sub_proyecto_1','Sub_proyecto_2','Sub_proyecto_3','Sistema','Destino','Cuenta','Proveedor'],
            ['Proyecto','Sub_proyecto_1','Categoria', 'Sub_categoria_1', 'Cuenta'])
        
        campos1 = list(map(str.lower, campos1))

        campos2 = st.multiselect(
            'Campos Gráfico 2 (el orden importa)',
            options=['Categoria','Sub_categoria_1','Proyecto','Sub_proyecto_1','Sub_proyecto_2','Sub_proyecto_3','Sistema','Destino','Cuenta','Proveedor'],
            default=['Categoria','Sub_categoria_1','Proyecto','Sub_proyecto_1','Cuenta'])
        
        campos2 = list(map(str.lower, campos2))

        submitted = st.form_submit_button(label = 'Submit')
    
    if submitted:
        data_proyectos = data_proyectos[
                            (data_proyectos.proyecto.isin(proyectos_elegidos))
                        ].sort_values(['fecha','id']).reset_index(drop=True).copy()

        data_proyectos[moneda] = data_proyectos[moneda].round(decimals=2)
        
        fig = px.treemap(data_proyectos, path=[px.Constant("Todos")] + campos1, values=moneda)
        fig.update_traces(root_color="lightgrey")
        fig.update_layout(margin = dict(t=50, l=25, r=25, b=25))
        fig.data[0].textinfo = 'label+text+value'
        st.subheader( ' --> '.join(map(str.title, campos1)) )
        st.plotly_chart(fig, use_container_width=True)

        fig = px.treemap(data_proyectos, path=[px.Constant("Todos")] + campos2, values=moneda)
        fig.update_traces(root_color="lightgrey")
        fig.update_layout(margin = dict(t=50, l=25, r=25, b=25))
        fig.data[0].textinfo = 'label+text+value'
        st.subheader( ' --> '.join(map(str.title, campos2)) )
        st.plotly_chart(fig, use_container_width=True)


    # Tabla

    st.subheader('Datos Seleccionados')

    tmp = data_proyectos[::-1].fillna('').copy().reset_index(drop=True)
    nombre = 'gasto (' + moneda + ')'
    tmp[nombre] = tmp[moneda]
    tmp = tmp[['id','fecha',nombre,'categoria','sub_categoria_1','sub_categoria_2','proyecto','sub_proyecto_1','sistema','cuenta','proveedor','detalle',
                'comprobante','site']]
    #tmp[nombre] = tmp[nombre].map('${:,.2f}'.format)
    tmp[nombre] = tmp[nombre].round(2)
    tmp['id'] = tmp.index
    
    gb = GridOptionsBuilder.from_dataframe(tmp)
    gb.configure_pagination()
    gb.configure_side_bar()
    gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc='sum', editable=True)
    gridOptions = gb.build()
    
    #gridOptions['columnDefs'] = [{'field':col, 'pivot':True, 'value':True} if col in ['categoria','sub_categoria_1'] else \
    #    {'field':col, 'pivot':False, 'value':True} for col in mayor.columns]
    AgGrid(tmp, gridOptions = gridOptions)#, enable_enterprise_modules=True)

#%% APORTES

def aportes(data, moneda, date_range):
    aportes = data[data.sub_categoria_1 == 'Inyección de Capital'].reset_index(drop=True).copy()
    aportes.detalle = aportes.detalle.fillna('')
    aportes = aportes[aportes.cuenta != 'montero incorporated'].reset_index(drop=True)
    aportes['tranche'] = 'NA'
    aportes.loc[aportes.detalle.str.contains('22m'), 'tranche'] = '22M'
    aportes.loc[aportes.detalle.str.contains('30m'), 'tranche'] = '30M'
    aportes.loc[aportes.detalle.str.contains('40m'), 'tranche'] = '40M'

    st.write('Total Aportes: USD {:,.0f}'.format(aportes.usd.sum()))
    
    tabla_aportes = pd.pivot_table(data=aportes, values='usd', index='month', columns='site', aggfunc=sum, fill_value=0)
    
    fig = go.Figure(data=[
                      go.Bar(
                          name=col,
                          x=tabla_aportes.index,
                          y=tabla_aportes[col])
                          for col in tabla_aportes
                     ])


    fig.update_layout(barmode='stack')
    
    fig.update_layout(title='<b>Historial de Aportes</b>')
    fig.update_yaxes(title_text='USD')
    st.plotly_chart(fig, use_container_width=True)


    pivot = pivot_ui(
        aportes.loc[
            aportes.fecha.between(date_range[0], date_range[1]),
            [
                'fecha',
                'destino',
                'cuenta',
                'detalle',
                'tranche',
                'site',
                'usd',
            ]
        ],
        rows=['tranche'],
        cols=['site'],
        vals=['usd'],
        aggregatorName='Sum',
        outfile_path='/tmp/pivottablejs.html'
        )
    st.header('Resumen de Aportes')
    st.write('Para el período ' + str(date_range[0]) + ' - ' + str(date_range[1]))
    with open(pivot.src) as pivot:
        components.html(pivot.read(), width=900, height=1000, scrolling=True)


#%% EQUIPO

def equipo(data, teams, moneda, date_range):
    payroll = pd.concat([data[data.sub_categoria_1.str.contains('Sala')].groupby('month').usd.sum(),
                     teams.sum(axis=1)], axis=1)
    payroll.columns = ['Payroll', 'Team Size']
    fig = make_subplots( specs = [[{'secondary_y':True}]] )

    fig.add_trace(
        go.Bar(
            x=payroll.index,
            y=payroll.Payroll,
            name='Payroll (left)'
        ),
        secondary_y=False
    )

    fig.add_trace(
        go.Scatter(
            x=payroll.index,
            y=payroll['Team Size'],
            name='Team Size (right)'
        ),
        secondary_y=True
    )

    fig.update_yaxes(title_text="US$ per month", secondary_y=False)
    fig.update_yaxes(title_text="Team Size", secondary_y=True)
    fig.update_layout(title='<b>Payroll y Equipo</b>')
    st.plotly_chart(fig, use_container_width=True)