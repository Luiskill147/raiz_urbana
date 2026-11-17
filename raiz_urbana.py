"""Código para análise de dados meteorológicos de duas estações do INMET em São Paulo, com foco em temperatura e precipitação horária.
Global Solution: differentiated problem solving
Curso:Engenharia de Software 
Turma:1ESPG
Integrantes: Eduardo Menzel - RM 570405
Luís Carlos - RM 569214 
Victor da Silva Souto - 571093"""
from pathlib import Path
from zipfile import ZipFile
from io import BytesIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PASTA = Path(__file__).resolve().parent
ARQUIVO = PASTA / '2026.zip'
ARQUIVO_RECORTADO = PASTA / 'dados_raiz_urbana_2026_10_25.csv'
CODIGOS = {'A701': 'Mirante', 'A771': 'Interlagos'}


def carregar_estacao(zip_inmet, codigo):
    nomes = [n for n in zip_inmet.namelist() if f'_{codigo}_' in n and n.lower().endswith('.csv')]
    if len(nomes) != 1:
        raise ValueError(f'Não foi encontrado um CSV único da estação {codigo} no ZIP.' )
    dados = pd.read_csv(BytesIO(zip_inmet.read(nomes[0])), sep=';',
                        skiprows=8, encoding='latin1', dtype=str )
    dados['data_hora'] = pd.to_datetime(dados['Data'] + ' ' + dados['Hora UTC'].str[:4],
                                        format='%Y/%m/%d %H%M', errors='coerce')
    colunas = {
        'temperatura': 'TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)' ,
        'chuva': 'PRECIPITAÇÃO TOTAL, HORÁRIO (mm)',
        'umidade': 'UMIDADE RELATIVA DO AR, HORARIA (%)'
    }
    for nome, coluna in colunas.items():
        dados[nome] = pd.to_numeric(dados[coluna].str.replace(',', '.', regex=False), errors='coerce' )
    return dados


def selecionar_dia(dados):
    # Critério transparente: maior precipitação conjunta entre os dias com
    # 24 medições válidas de chuva e temperatura em ambas as estações.
    estatisticas = []
    agrupamentos = {c: {dia: grupo for dia, grupo in df.groupby(df.data_hora.dt.date)}
                     for c, df in dados.items()}
    for dia in sorted(set(agrupamentos['A701']) & set(agrupamentos['A771'])):
        registros = [agrupamentos[c][dia] for c in CODIGOS]
        if all(len(df) == 24 and df.temperatura.notna().all()
               and df.chuva.notna().all() for df in registros):
            estatisticas.append((sum(float(df.chuva.sum()) for df in registros), dia))
    if not estatisticas:
        raise ValueError('Não há dia com dados completos nas duas estações.')
    _, dia = max(estatisticas)
    return dia, {c: agrupamentos[c][dia].sort_values('data_hora').reset_index(drop=True)
                 for c in CODIGOS}


def graficos(dia, diarios, coef):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for codigo, nome in CODIGOS.items():
        ax.plot(range(24), diarios[codigo].temperatura, marker='o', markersize=3,
                linewidth=1.6, label=f'{nome} ({codigo})')
    ax.set(title=f'Temperatura horária • {dia:%d/%m/%Y}', xlabel='Hora (UTC)',
           ylabel='Temperatura (°C)', xticks=range(0, 24, 2))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PASTA / 'raiz_urbana_temperatura.png', dpi=190)
    plt.close(fig)

    horas = np.arange(6, 13)
    temperaturas = diarios['A701'].temperatura.to_numpy()[6:13]
    u = np.linspace(0, 6, 150)
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    ax.scatter(horas, temperaturas, label='Temperatura observada (INMET)', zorder=5)
    ax.plot(u + 6, np.polyval(coef, u), label='Modelo quadrático aproximado')
    ax.set(title='Ajuste matemático • Mirante • 6h–12h UTC', xlabel='Hora (UTC)',
           ylabel='Temperatura (°C)', xticks=range(6, 13))
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(PASTA / 'raiz_urbana_ajuste.png', dpi=190)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for codigo, nome in CODIGOS.items():
        ax.step(np.arange(25), np.r_[0, diarios[codigo].chuva.cumsum()],
                where='post', linewidth=1.8, label=f'{nome} ({codigo})')
    ax.set(title=f'Precipitação acumulada • {dia:%d/%m/%Y}', xlabel='Hora (UTC)',
           ylabel='Precipitação acumulada (mm)', xticks=range(0, 25, 2))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PASTA / 'raiz_urbana_precipitacao.png', dpi=190)
    plt.close(fig)


def main():
    if ARQUIVO.exists():
        with ZipFile(ARQUIVO) as z:
            dados = {codigo: carregar_estacao(z, codigo) for codigo in CODIGOS}
        dia, diarios = selecionar_dia(dados)
    elif ARQUIVO_RECORTADO.exists():
        recorte = pd.read_csv(ARQUIVO_RECORTADO, parse_dates=['data_hora'])
        diarios = {codigo: recorte[recorte.estacao == codigo].sort_values('data_hora').reset_index(drop=True)
                   for codigo in CODIGOS}
        if any(len(df) != 24 for df in diarios.values()):
            raise ValueError('O CSV recortado deve ter 24 registros por estação.')
        dia = diarios['A701'].data_hora.iloc[0].date()
    else:
        raise FileNotFoundError('Coloque 2026.zip OU dados_raiz_urbana_2026_10_25.csv na pasta do código.')

    # u = hora UTC - 6, com 0 <= u <= 6.
    u = np.arange(7)
    valores = diarios['A701'].temperatura.to_numpy()[6:13]
    coef = np.polyfit(u, valores, 2)
    a, b, c = coef
    valores_estimados = np.polyval(coef, u)
    r2 = 1 - np.sum((valores - valores_estimados) ** 2) / np.sum((valores - valores.mean()) ** 2)
    u_min = -b / (2 * a)
    chuva_mirante = diarios['A701'].chuva.sum()
    chuva_interlagos = diarios['A771'].chuva.sum()
    graficos(dia, diarios, coef)

    print(f'Data selecionada: {dia:%d/%m/%Y}')
    for codigo, nome in CODIGOS.items():
        df = diarios[codigo]
        print(f'{nome}: mínima {df.temperatura.min():.1f}°C; máxima {df.temperatura.max():.1f}°C; '
              f'chuva acumulada {df.chuva.sum():.1f} mm')
    print(f'Modelo: T(u) = {a:.6f}u² + ({b:.6f})u + {c:.6f}; 0 ≤ u ≤ 6')
    print(f'R² aproximado: {r2:.4f}')
    print(f'Limite u→3 = T(3) = {np.polyval(coef, 3):.3f}°C')
    print(f'Derivada T\'(3) = {2*a*3+b:.3f} °C/h')
    print(f'Mínimo do modelo: u={u_min:.3f} h após 6h UTC, '
          f'T={np.polyval(coef,u_min):.3f}°C')
    print(f'Chuva Mirante - Interlagos = {chuva_mirante-chuva_interlagos:.1f} mm')
    print('Gráficos PNG gerados na pasta do programa.')


if __name__ == '__main__':
    main()
