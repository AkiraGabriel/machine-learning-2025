import streamlit as st
import pandas as pd

st.markdown("# Descubra a Felicidade")

col1, col2 = st.columns(2)

plataforma = ['LinkedIn','Twitch', 'YouTube', 'Instagram', 'Amigos','Twitter / X', 'Outra rede social']
qtd_cursos = ['0', '2', '1', '3', 'Mais que 3']
estados_lista = ['MG', 'SC', 'SP', 'CE', 'PE', 'RJ', 'AM', 'PR', 'BA', 'PA', 'MT',
'RS', 'DF', 'RN', 'ES', 'PB', 'GO', 'MA']
senioridade = ['Iniciante', 'Júnior', 'Pleno', 'Sênior', 'Especialista','Coordenação',  'Gerência',  'Diretoria', 'C-Level']
areas_formacao = ['Exatas', 'Biológicas', 'Humanas']
with col1:
    video_games = st.radio("Curte games?", ["Sim", 'Não'])
    futebol = st.radio("Curte futebol?", ["Sim", 'Não'])
    livros = st.radio("Curte livros?", ["Sim", 'Não'])
    redes = st.selectbox("Como conheceu o Téo?", options=plataforma)
    cursos = st.selectbox('Quantos cursos acompanhou do Téo Me Why?', options=qtd_cursos)
    formacao = st.selectbox("Área de Formação", options=areas_formacao)
    cadeira = st.selectbox("Posição da cadeira (senioridade)", options=senioridade)

with col2:
    tabuleiro = st.radio("Curte jogos de tabuleiro?", ["Sim", 'Não'])
    f1 = st.radio("Curte jogos de fórmula 1?", ["Sim", 'Não'])
    MMA = st.radio("Curte jogos de MMA?", ["Sim", 'Não'])
    idade = st.number_input("Qual sua idade?", 18, 100)
    estado = st.selectbox('Estado que mora atualmente', options=sorted(estados_lista))
    tempo_area = st.radio("Tempo que atua na área de dados", ['Não atuo', 'De 1 ano a 2 anos', 'Mais de 4 anos'])

data={
    'Como conheceu o Téo Me Why?': redes,
    'Quantos cursos acompanhou do Téo Me Why?':cursos,
    'Curte games?':video_games,
    'Curte futebol?':futebol,
    'Curte livros?':livros,
    'Curte jogos de tabuleiro?':tabuleiro,
    'Curte jogos de fórmula 1?':f1,
    'Curte jogos de MMA?':MMA,
    'Idade':idade,
    'Estado que mora atualmente':estado,
    'Área de Formação':formacao,
    'Tempo que atua na área de dados':tempo_area,
    'Posição da cadeira (senioridade)':cadeira
}

df = pd.DataFrame([data])
st.data_editor(df)