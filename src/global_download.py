import logging
from data_repository import LocalCSVStorage
from data_cleaner import DataCleaner
import os
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

assets_portfolio = {
    'Сбербанк': [{'secid': 'RU0009029540', 'engine': 'stock', 'market': 'shares', 'board': 'EQBR'},         # 2000-10-12:2006-07-20 -листинг+дивы
                 {'secid': 'SBER-2007', 'engine': 'stock', 'market': 'shares', 'board': 'EQBR'},            # 2006-08-04:2007-07-17
                 {'secid': 'SBER03', 'engine': 'stock', 'market': 'shares', 'board': 'EQBR'},               # 2007-07-20:2011-11-18
                 {'secid': 'SBER', 'engine': 'stock', 'market': 'shares', 'board': 'EQBR'},                 # 2011-11-21:2013-08-30
                 {'secid': 'SBER', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'}],                # 2013-03-25:наши дни
    'Т-Технологии': [{'secid': 'TCSG', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'},             # 2019-10-28:2024-11-27 -листинг+дивы
                     {'secid': 'T', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'}],               # 2024-11-28:наши дни
    'Яндекс': [{'secid': 'YNDX', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'},                   # 2014-06-04:2024-06-14 -листинг+дивы
               {'secid': 'YDEX', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'}],                  # 2024-07-24:наши дни
    'РусАгро': [{'secid': 'AGRO', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'},                  # 2014-12-01:2024-12-02 -листинг+дивы
                {'secid': 'RAGR', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'}],                 # 2025-02-17:наши дни
    'X5 Group': [{'secid': 'FIVE', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'},                 # 2018-02-01:2024-04-03 -листинг+дивы
                 {'secid': 'X5', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'}],                  # 2025-01-09:наши дни
    'МТС': [{'secid': 'RU14MTSG1006', 'engine': 'stock', 'market': 'shares', 'board': 'EQNE'},              # 2003-10-15:2003-12-10 -листинг+дивы
            {'secid': 'RU14MTSG1006', 'engine': 'stock', 'market': 'shares', 'board': 'EQNL'},              # 2003-12-11:2004-02-10
            {'secid': 'MTSI', 'engine': 'stock', 'market': 'shares', 'board': 'EQNL'},                      # 2004-02-11:2010-10-20
            {'secid': 'MTSI', 'engine': 'stock', 'market': 'shares', 'board': 'EQBR'},                      # 2010-10-21:2011-11-18
            {'secid': 'MTSS', 'engine': 'stock', 'market': 'shares', 'board': 'EQBR'},                      # 2011-11-21:2013-08-30
            {'secid': 'MTSS', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'}],                     # 2013-07-08:наши дни
    'Ростелеком': [{'secid': 'RU0008943394', 'engine': 'stock', 'market': 'shares', 'board': 'EQBR'},       # 1997-03-24:2003-09-30 -листинг+дивы
                   {'secid': 'RTKM', 'engine': 'stock', 'market': 'shares', 'board': 'EQBR'},               # 2003-10-01:2013-08-30
                   {'secid': 'RTKM', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'}],              # 2013-07-09:наши дни
    'Ростелеком П': [{'secid': 'RU0009046700', 'engine': 'stock', 'market': 'shares', 'board': 'EQBR'},     # 1997-07-15:2003-09-30 -листинг+дивы
                     {'secid': 'RTKMP', 'engine': 'stock', 'market': 'shares', 'board': 'EQBR'},            # 2003-10-01:2013-08-30
                     {'secid': 'RTKMP', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'}],           # 2013-08-01:наши дни
    'Астра': [{'secid': 'ASTR', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'}],                   # 2023-10-13:наши дни   +листинг+дивы
    'Диасофт': [{'secid': 'DIAS', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'}],                 # 2024-02-13:наши дни   +листинг+дивы
    'Позитив': [{'secid': 'POSI', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'}],                 # 2021-12-17:наши дни   +листинг+дивы
    'ИнтерРАО': [{'secid': 'IUES', 'engine': 'stock', 'market': 'shares', 'board': 'EQNL'},                 # 2009-12-01:2011-09-26 -листинг+дивы
                 {'secid': 'IUES', 'engine': 'stock', 'market': 'shares', 'board': 'EQBR'},                 # 2011-09-27:2011-11-18
                 {'secid': 'IRAO', 'engine': 'stock', 'market': 'shares', 'board': 'EQBR'},                 # 2011-11-21:2013-08-30
                 {'secid': 'IRAO', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'}],                # 2013-07-11:наши дни
    'Юнипро': [{'secid': 'OGK4', 'engine': 'stock', 'market': 'shares', 'board': 'EQNL'},                   # 2007-05-24:2011-11-18 -листинг+дивы
               {'secid': 'EONR', 'engine': 'stock', 'market': 'shares', 'board': 'EQNL'},                   # 2011-11-21:2013-08-30
               {'secid': 'EONR', 'engine': 'stock', 'market': 'shares', 'board': 'TQNL'},                   # 2013-07-11:2014-06-06
               {'secid': 'EONR', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'},                   # 2014-06-09:2016-06-30
               {'secid': 'UPRO', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'}],                  # 2016-07-01:наши дни
    'ВК': [{'secid': 'MAIL', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'},                       # 2020-07-02:2021-12-13 -листинг-дивы
           {'secid': 'VKCO', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'}],                      # 2021-12-14:наши дни
    'АФК-Система': [{'secid': 'AFKC', 'engine': 'stock', 'market': 'shares', 'board': 'EQLV'},              # 2007-12-12:2008-07-03 -листинг+дивы
                    {'secid': 'AFKC', 'engine': 'stock', 'market': 'shares', 'board': 'EQNL'},              # 2008-07-04:2011-11-18
                    {'secid': 'AFKS', 'engine': 'stock', 'market': 'shares', 'board': 'EQNL'},              # 2011-11-21:2013-08-30
                    {'secid': 'AFKS', 'engine': 'stock', 'market': 'shares', 'board': 'TQNL'},              # 2013-07-08:2014-04-07
                    {'secid': 'AFKS', 'engine': 'stock', 'market': 'shares', 'board': 'TQBS'},              # 2014-04-08:2014-06-06
                    {'secid': 'AFKS', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'}],             # 2014-06-09:наши дни
    'ЭсЭфАй': [{'secid': 'EPLN', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'},                   # 2015-12-11:2017-12-29 -листинг+дивы
               {'secid': 'SFIN', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'}],                  # 2018-01-03:наши дни
    'Whoosh': [{'secid': 'WUSH', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'}],                  # 2022-12-14:наши дни   +листинг+дивы
    'Мать и Дитя': [{'secid': 'MDMG', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'}],             # 2020-11-09:наши дни   -листинг+дивы
    'Газпром': [{'secid': 'GAZP', 'engine': 'stock', 'market': 'shares', 'board': 'EQNE'},                  # 2006-01-23:2013-08-30 -листинг+дивы
                {'secid': 'GAZP', 'engine': 'stock', 'market': 'shares', 'board': 'TQNE'},                  # 2013-03-25:2013-12-25
                {'secid': 'GAZP', 'engine': 'stock', 'market': 'shares', 'board': 'TQBS'},                  # 2013-12-26:2014-06-06
                {'secid': 'GAZP', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'}],                 # 2014-06-09:наши дни
    'Газпром Нефть': [{'secid': 'RU14SIBN1003', 'engine': 'stock', 'market': 'shares', 'board': 'EQBS'},    # 1999-09-06:2002-08-12 -листинг+дивы
                      {'secid': 'RU14SIBN1003', 'engine': 'stock', 'market': 'shares', 'board': 'EQBR'},    # 2002-08-13:2003-07-09
                      {'secid': 'SIBN', 'engine': 'stock', 'market': 'shares', 'board': 'EQBR'},            # 2003-07-10:2003-12-16
                      {'secid': 'SIBN', 'engine': 'stock', 'market': 'shares', 'board': 'EQNL'},            # 2003-12-17:2010-02-03
                      {'secid': 'SIBN', 'engine': 'stock', 'market': 'shares', 'board': 'EQNE'},            # 2010-02-04:2013-08-30
                      {'secid': 'SIBN', 'engine': 'stock', 'market': 'shares', 'board': 'TQNE'},            # 2013-08-22:2014-06-06
                      {'secid': 'SIBN', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'}],           # 2014-06-09:наши дни
    'Лукойл': [{'secid': 'RU0009024277', 'engine': 'stock', 'market': 'shares', 'board': 'EQBR'},           # 1997-04-25:2003-08-19 -листинг+дивы
               {'secid': 'LKOH', 'engine': 'stock', 'market': 'shares', 'board': 'EQBR'},                   # 2003-08-20:2013-08-30
               {'secid': 'LKOH', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'}],                  # 2013-03-25:наши дни
    'Новатэк': [{'secid': 'NOTK', 'engine': 'stock', 'market': 'shares', 'board': 'EQBS'},                  # 2009-08-07:2009-10-08 -листинг+дивы
                {'secid': 'NOTK', 'engine': 'stock', 'market': 'shares', 'board': 'EQBR'},                  # 2009-10-09:2011-11-18
                {'secid': 'NVTK', 'engine': 'stock', 'market': 'shares', 'board': 'EQBR'},                  # 2011-11-21:2013-08-30
                {'secid': 'NVTK', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'}],                 # 2013-04-10:наши дни
    'Роснефть': [{'secid': 'ROSN', 'engine': 'stock', 'market': 'shares', 'board': 'EQLV'},                 # 2006-07-19:2006-12-19 -листинг+дивы
                 {'secid': 'ROSN', 'engine': 'stock', 'market': 'shares', 'board': 'EQNL'},                 # 2006-12-20:2013-08-30
                 {'secid': 'ROSN', 'engine': 'stock', 'market': 'shares', 'board': 'TQNL'},                 # 2013-03-25:2014-05-19
                 {'secid': 'ROSN', 'engine': 'stock', 'market': 'shares', 'board': 'TQBS'},                 # 2014-05-20:2014-06-06
                 {'secid': 'ROSN', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'}],                # 2014-06-09:наши дни
    'Сургут П': [{'secid': 'RU0009029524', 'engine': 'stock', 'market': 'shares', 'board': 'EQNL'},         # 1997-11-26:2003-01-31 -листинг+дивы
                 {'secid': 'RU0009029524', 'engine': 'stock', 'market': 'shares', 'board': 'EQNE'},         # 2003-02-03:2003-07-30
                 {'secid': 'SNGSP', 'engine': 'stock', 'market': 'shares', 'board': 'EQNE'},                # 2003-07-31:2011-12-09
                 {'secid': 'SNGSP', 'engine': 'stock', 'market': 'shares', 'board': 'EQNL'},                # 2011-12-12:2013-08-30
                 {'secid': 'SNGSP', 'engine': 'stock', 'market': 'shares', 'board': 'TQNL'},                # 2013-07-10:2014-06-06
                 {'secid': 'SNGSP', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'}],               # 2014-06-09:наши дни
    'Татнефть': [{'secid': 'RU14TATN3006', 'engine': 'stock', 'market': 'shares', 'board': 'EQBR'},         # 2002-01-04:2011-11-18 -листинг+дивы
                 {'secid': 'TATN', 'engine': 'stock', 'market': 'shares', 'board': 'EQBR'},                 # 2011-11-21:2013-08-30
                 {'secid': 'TATN', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'}],                # 2013-07-10:наши дни
    'Алроса': [{'secid': 'ALRS', 'engine': 'stock', 'market': 'shares', 'board': 'EQNE'},                   # 2011-11-29:2012-04-20 +листинг+дивы
               {'secid': 'ALRS', 'engine': 'stock', 'market': 'shares', 'board': 'EQBS'},                   # 2012-04-23:2012-10-01
               {'secid': 'ALRS', 'engine': 'stock', 'market': 'shares', 'board': 'EQBR'},                   # 2012-10-02:2013-08-30
               {'secid': 'ALRS', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'}],                  # 2013-09-02:наши дни
    'Мечел': [{'secid': 'MTLR', 'engine': 'stock', 'market': 'shares', 'board': 'EQLV'},                    # 2008-12-26:2009-03-18 -листинг+дивы
              {'secid': 'MTLR', 'engine': 'stock', 'market': 'shares', 'board': 'EQBR'},                    # 2009-03-19:2013-08-30
              {'secid': 'MTLR', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'}],                   # 2013-07-08:наши дни
    'ММК': [{'secid': 'MAGN', 'engine': 'stock', 'market': 'shares', 'board': 'EQNE'},                      # 2006-01-18:2006-07-03 -листинг+дивы
            {'secid': 'MAGN', 'engine': 'stock', 'market': 'shares', 'board': 'EQNL'},                      # 2006-07-04:2013-08-30
            {'secid': 'MAGN', 'engine': 'stock', 'market': 'shares', 'board': 'TQNL'},                      # 2013-07-11:2014-06-06
            {'secid': 'MAGN', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'}],                     # 2014-06-09:наши дни
    'НЛМК': [{'secid': 'NLMK', 'engine': 'stock', 'market': 'shares', 'board': 'EQNE'},                     # 2006-04-18:2006-08-08 -листинг+дивы
             {'secid': 'NLMK', 'engine': 'stock', 'market': 'shares', 'board': 'EQNL'},                     # 2006-08-09:2013-08-30
             {'secid': 'NLMK', 'engine': 'stock', 'market': 'shares', 'board': 'TQNL'},                     # 2013-07-08:2014-06-06
             {'secid': 'NLMK', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'}],                    # 2014-06-09:наши дни
    'Норникель': [{'secid': 'RU14GMKN0507', 'engine': 'stock', 'market': 'shares', 'board': 'EQNL'},        # 2003-02-03:2006-12-25 -листинг+дивы
                  {'secid': 'GMKN', 'engine': 'stock', 'market': 'shares', 'board': 'EQNL'},                # 2006-12-26:2010-04-27
                  {'secid': 'GMKN', 'engine': 'stock', 'market': 'shares', 'board': 'EQBR'},                # 2010-04-28:2011-08-11
                  {'secid': 'GMKN', 'engine': 'stock', 'market': 'shares', 'board': 'EQBS'},                # 2011-08-12:2013-08-30
                  {'secid': 'GMKN', 'engine': 'stock', 'market': 'shares', 'board': 'TQBS'},                # 2013-03-25:2014-06-06
                  {'secid': 'GMKN', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'}],               # 2014-06-09:наши дни
    'Полюс': [{'secid': 'PLZL', 'engine': 'stock', 'market': 'shares', 'board': 'EQNE'},                    # 2006-05-12:2013-08-30 -листинг+дивы
              {'secid': 'PLZL', 'engine': 'stock', 'market': 'shares', 'board': 'TQNE'},                    # 2013-09-02:2014-06-06
              {'secid': 'PLZL', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'}],                   # 2014-06-09:наши дни
    'Распадская': [{'secid': 'RASP', 'engine': 'stock', 'market': 'shares', 'board': 'EQLV'},               # 2006-11-14:2007-03-29 -листинг+дивы
                   {'secid': 'RASP', 'engine': 'stock', 'market': 'shares', 'board': 'EQNL'},               # 2007-03-30:2013-08-30
                   {'secid': 'RASP', 'engine': 'stock', 'market': 'shares', 'board': 'TQNL'},               # 2013-07-08:2014-06-06
                   {'secid': 'RASP', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'}],              # 2014-06-09:наши дни
    'РусАл': [{'secid': 'RUAL', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'}],                   # 2015-03-30:наши дни   -листинг+дивы
    'СеверСталь': [{'secid': 'CHMF', 'engine': 'stock', 'market': 'shares', 'board': 'EQNL'},               # 2005-06-22:2013-08-30 -листинг+дивы
                   {'secid': 'CHMF', 'engine': 'stock', 'market': 'shares', 'board': 'TQNL'},               # 2013-04-03:2014-04-07
                   {'secid': 'CHMF', 'engine': 'stock', 'market': 'shares', 'board': 'TQBS'},               # 2014-04-08:2014-06-06
                   {'secid': 'CHMF', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'}],              # 2014-06-09:наши дни
    'Сегежа': [{'secid': 'SGZH', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'}],                  # 2021-04-28:наши дни   +листинг+дивы
    'ФосАгро': [{'secid': 'PHOR', 'engine': 'stock', 'market': 'shares', 'board': 'EQLV'},                  # 2011-07-18:2012-11-13 -листинг+дивы
                {'secid': 'PHOR', 'engine': 'stock', 'market': 'shares', 'board': 'EQNL'},                  # 2012-11-14:2013-02-27
                {'secid': 'PHOR', 'engine': 'stock', 'market': 'shares', 'board': 'EQBR'},                  # 2013-02-28:2013-08-30
                {'secid': 'PHOR', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'}],                 # 2013-09-02:наши дни
    'ЮжУралЗолото': [{'secid': 'UGLD', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'}],            # 2023-11-22:наши дни   +листинг+дивы
    'Хэдхантер': [{'secid': 'HHRU', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'},                # 2020-09-25:2024-08-09 -листинг+дивы
                  {'secid': 'HEAD', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'}],               # 2024-09-26:наши дни
    'Пик': [{'secid': 'PIKK', 'engine': 'stock', 'market': 'shares', 'board': 'EQLV'},                      # 2007-06-29:2007-12-19 -листинг+дивы
            {'secid': 'PIKK', 'engine': 'stock', 'market': 'shares', 'board': 'EQNL'},                      # 2007-12-20:2011-09-06
            {'secid': 'PIKK', 'engine': 'stock', 'market': 'shares', 'board': 'EQBR'},                      # 2011-09-07:2013-08-30
            {'secid': 'PIKK', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'}],                     # 2013-09-02:наши дни
    'Самолет': [{'secid': 'SMLT', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'}],                 # 2020-10-29:наши дни   +листинг+дивы
    'Эталон': [{'secid': 'ETLN', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'}],                  # 2020-02-03:наши дни   -листинг+дивы
    'М-Видео': [{'secid': 'MVID', 'engine': 'stock', 'market': 'shares', 'board': 'EQLI'},                  # 2007-11-06:2008-04-07 -листинг+дивы
                {'secid': 'MVID', 'engine': 'stock', 'market': 'shares', 'board': 'EQNL'},                  # 2008-04-08:2013-08-30
                {'secid': 'MVID', 'engine': 'stock', 'market': 'shares', 'board': 'TQNL'},                  # 2013-09-02:2013-10-30
                {'secid': 'MVID', 'engine': 'stock', 'market': 'shares', 'board': 'TQBS'},                  # 2013-10-31:2014-06-06
                {'secid': 'MVID', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'}],                 # 2014-06-09:наши дни
    'Хэндерсон': [{'secid': 'HNFG', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'}],               # 2023-11-02:наши дни   +листинг+дивы
    'Озон': [{'secid': 'OZON', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'}],                    # 2020-11-24:наши дни   -листинг+дивы
    'Аэрофлот': [{'secid': 'RU0009062285', 'engine': 'stock', 'market': 'shares', 'board': 'EQNL'},         # 2000-03-09:2001-04-27 -листинг+дивы
                 {'secid': 'RU0009062285', 'engine': 'stock', 'market': 'shares', 'board': 'EQBS'},         # 2001-04-28:2004-02-12
                 {'secid': 'AFLT', 'engine': 'stock', 'market': 'shares', 'board': 'EQBR'},                 # 2007-08-21:2013-05-07
                 {'secid': 'AFLT', 'engine': 'stock', 'market': 'shares', 'board': 'EQBS'},                 # 2004-02-13:2007-08-20 + 2013-05-08:2013-08-30
                 {'secid': 'AFLT', 'engine': 'stock', 'market': 'shares', 'board': 'TQBS'},                 # 2013-07-23:2013-10-30
                 {'secid': 'AFLT', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'}],                # 2013-10-31:наши дни
    'Белуга': [{'secid': 'SYNG', 'engine': 'stock', 'market': 'shares', 'board': 'EQNE'},                   # 2007-11-21:2010-01-21 -листинг+дивы
               {'secid': 'SYNG', 'engine': 'stock', 'market': 'shares', 'board': 'EQNL'},                   # 2010-01-22:2013-08-30
               {'secid': 'SYNG', 'engine': 'stock', 'market': 'shares', 'board': 'TQNL'},                   # 2013-09-02:2014-06-05
               {'secid': 'SYNG', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'},                   # 2014-06-09:2017-07-31
               {'secid': 'BELU', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'}],                  # 2017-08-01:наши дни
    'Инарктика': [{'secid': 'RSEA', 'engine': 'stock', 'market': 'shares', 'board': 'EQNE'},                # 2010-04-20:2010-08-12 -листинг+дивы
                  {'secid': 'RSEA', 'engine': 'stock', 'market': 'shares', 'board': 'EQNL'},                # 2010-08-13:2013-08-30
                  {'secid': 'RSEA', 'engine': 'stock', 'market': 'shares', 'board': 'TQNL'},                # 2013-09-02:2014-06-06
                  {'secid': 'RSEA', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'},                # 2014-06-09:2015-06-09
                  {'secid': 'AQUA', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'}],               # 2015-06-10:наши дни
    'Магнит': [{'secid': 'MGNT', 'engine': 'stock', 'market': 'shares', 'board': 'EQNE'},                   # 2006-05-16:2007-12-26 -листинг+дивы
               {'secid': 'MGNT', 'engine': 'stock', 'market': 'shares', 'board': 'EQNL'},                   # 2007-12-27:2009-08-12
               {'secid': 'MGNT', 'engine': 'stock', 'market': 'shares', 'board': 'EQBS'},                   # 2009-08-13:2011-01-11
               {'secid': 'MGNT', 'engine': 'stock', 'market': 'shares', 'board': 'EQBR'},                   # 2011-01-12:2013-08-30
               {'secid': 'MGNT', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'}],                  # 2013-07-08:наши дни
    'Черкизово': [{'secid': 'GCHE', 'engine': 'stock', 'market': 'shares', 'board': 'EQNE'},                # 2008-05-19:2008-08-21 -листинг+дивы
                  {'secid': 'GCHE', 'engine': 'stock', 'market': 'shares', 'board': 'EQNL'},                # 2008-08-25:2010-04-01
                  {'secid': 'GCHE', 'engine': 'stock', 'market': 'shares', 'board': 'EQBS'},                # 2010-04-02:2012-03-01
                  {'secid': 'GCHE', 'engine': 'stock', 'market': 'shares', 'board': 'EQBR'},                # 2012-03-02:2013-08-30
                  {'secid': 'GCHE', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'}],               # 2013-09-02:наши дни
    'Лента': [{'secid': 'LENT', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'}],                   # 2021-12-14:наши дни   -листинг-дивы
    'НМТП': [{'secid': 'NMTP', 'engine': 'stock', 'market': 'shares', 'board': 'EQLV'},                     # 2007-11-08:2008-05-21 -листинг+дивы
             {'secid': 'NMTP', 'engine': 'stock', 'market': 'shares', 'board': 'EQNE'},                     # 2008-05-22:2008-07-31
             {'secid': 'NMTP', 'engine': 'stock', 'market': 'shares', 'board': 'EQNL'},                     # 2008-08-01:2013-08-30
             {'secid': 'NMTP', 'engine': 'stock', 'market': 'shares', 'board': 'TQNL'},                     # 2013-09-02:2014-06-06
             {'secid': 'NMTP', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'}],                    # 2014-06-09:наши дни
    'СовкомФлот': [{'secid': 'FLOT', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'}],              # 2020-10-07:наши дни   +листинг+дивы
    'НКХП': [{'secid': 'NKHP', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'}],                    # 2015-12-15:наши дни   +листинг+дивы
    'Транснефть П': [{'secid': 'RU14TRNF1015', 'engine': 'stock', 'market': 'shares', 'board': 'EQNL'},     # 2002-05-22:2008-01-11 -листинг+дивы
                     {'secid': 'TRNFP', 'engine': 'stock', 'market': 'shares', 'board': 'EQNL'},            # 2008-01-14:2013-08-30
                     {'secid': 'TRNFP', 'engine': 'stock', 'market': 'shares', 'board': 'TQNL'},            # 2013-03-25:2013-12-13
                     {'secid': 'TRNFP', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'}],           # 2013-12-16:наши дни
    'Банк СПБ': [{'secid': 'BSPB', 'engine': 'stock', 'market': 'shares', 'board': 'EQLV'},                 # 2008-03-19:2008-04-07 -листинг+дивы
                 {'secid': 'BSPB', 'engine': 'stock', 'market': 'shares', 'board': 'EQNL'},                 # 2008-04-08:2013-08-30
                 {'secid': 'BSPB', 'engine': 'stock', 'market': 'shares', 'board': 'TQNL'},                 # 2013-09-02:2014-06-06
                 {'secid': 'BSPB', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'}],                # 2014-06-09:наши дни
    'ВТБ': [{'secid': 'VTBR', 'engine': 'stock', 'market': 'shares', 'board': 'EQLV'},                      # 2007-05-28:2007-10-29 -листинг+дивы
            {'secid': 'VTBR', 'engine': 'stock', 'market': 'shares', 'board': 'EQNL'},                      # 2007-10-30:2013-02-27
            {'secid': 'VTBR', 'engine': 'stock', 'market': 'shares', 'board': 'EQBR'},                      # 2013-02-28:2013-08-30
            {'secid': 'VTBR', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'}],                     # 2013-03-25:наши дни
    'Европлан': [{'secid': 'LEAS', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'}],                # 2024-03-29:наши дни   +листинг+дивы(c NaN)
    'МосБиржа': [{'secid': 'MOEX', 'engine': 'stock', 'market': 'shares', 'board': 'EQLV'},                 # 2013-02-15:2013-04-03 -листинг+дивы
                 {'secid': 'MOEX', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'}],                # 2013-04-04:наши дни
    'Ренессанс': [{'secid': 'RENI', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'}],               # 2021-10-21:наши дни   +листинг+дивы
    'Совкомбанк': [{'secid': 'SVCB', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'}],              # 2023-12-15:наши дни   +листинг+дивы
    'СПБ Биржа': [{'secid': 'SPBE', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'}],               # 2021-11-19:наши дни   -листинг-дивы
    'КазОргСинтез': [{'secid': 'KZOS', 'engine': 'stock', 'market': 'shares', 'board': 'EQNE'},             # 2011-12-05:2013-08-30 -листинг+дивы
                     {'secid': 'KZOS', 'engine': 'stock', 'market': 'shares', 'board': 'TQNE'},             # 2013-09-02:2014-06-06
                     {'secid': 'KZOS', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'}],            # 2014-06-09:наши дни
    'НижКамНефХим': [{'secid': 'NKNC', 'engine': 'stock', 'market': 'shares', 'board': 'EQNE'},             # 2008-04-07:2009-07-23 -листинг+дивы
                     {'secid': 'NKNC', 'engine': 'stock', 'market': 'shares', 'board': 'EQBS'},             # 2009-07-24:2013-08-30
                     {'secid': 'NKNC', 'engine': 'stock', 'market': 'shares', 'board': 'TQBS'},             # 2013-09-02:2014-06-06
                     {'secid': 'NKNC', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'}],            # 2014-06-09:наши дни
    'НижКамНефХим П': [{'secid': 'NKNCP', 'engine': 'stock', 'market': 'shares', 'board': 'EQNE'},          # 2008-04-18:2009-07-23 -листинг+дивы
                       {'secid': 'NKNCP', 'engine': 'stock', 'market': 'shares', 'board': 'EQBS'},          # 2009-07-24:2010-04-01
                       {'secid': 'NKNCP', 'engine': 'stock', 'market': 'shares', 'board': 'EQBS'},          # 2010-04-02:2013-08-30
                       {'secid': 'NKNCP', 'engine': 'stock', 'market': 'shares', 'board': 'TQBS'},          # 2013-09-02:2014-06-06
                       {'secid': 'NKNCP', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'}],         # 2014-06-09:наши дни
    'Россети МР': [{'secid': 'MSRS', 'engine': 'stock', 'market': 'shares', 'board': 'EQNE'},               # 2006-02-10:2006-07-06 -листинг+дивы
                   {'secid': 'MSRS', 'engine': 'stock', 'market': 'shares', 'board': 'EQNL'},               # 2006-07-07:2008-05-06
                   {'secid': 'MSRS', 'engine': 'stock', 'market': 'shares', 'board': 'EQBS'},               # 2008-05-07:2009-02-18
                   {'secid': 'MSRS', 'engine': 'stock', 'market': 'shares', 'board': 'EQBR'},               # 2009-02-19:2013-08-30
                   {'secid': 'MSRS', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'}],              # 2013-09-02:наши дни
    'Россети Центр': [{'secid': 'MRKC', 'engine': 'stock', 'market': 'shares', 'board': 'EQNE'},            # 2008-08-18:2009-04-13 -листинг+дивы
                      {'secid': 'MRKC', 'engine': 'stock', 'market': 'shares', 'board': 'EQNL'},            # 2009-04-14:2011-08-15
                      {'secid': 'MRKC', 'engine': 'stock', 'market': 'shares', 'board': 'EQBS'},            # 2011-08-16:2012-10-19
                      {'secid': 'MRKC', 'engine': 'stock', 'market': 'shares', 'board': 'EQBR'},            # 2012-10-22:2013-08-30
                      {'secid': 'MRKC', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'}],           # 2013-09-02:наши дни
    'Россети ЦиП': [{'secid': 'MRKP', 'engine': 'stock', 'market': 'shares', 'board': 'EQNE'},              # 2008-08-05:2008-09-23 -листинг+дивы
                    {'secid': 'MRKP', 'engine': 'stock', 'market': 'shares', 'board': 'EQNL'},              # 2008-09-24:2011-05-17
                    {'secid': 'MRKP', 'engine': 'stock', 'market': 'shares', 'board': 'EQBS'},              # 2011-05-18:2012-02-24
                    {'secid': 'MRKP', 'engine': 'stock', 'market': 'shares', 'board': 'EQBR'},              # 2012-02-27:2013-08-30
                    {'secid': 'MRKP', 'engine': 'stock', 'market': 'shares', 'board': 'TQBS'},              # 2014-04-08:2014-06-06
                    {'secid': 'MRKP', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'}],             # 2013-09-02:2014-04-07 + 2014-06-09:наши дни
    'Россети C-З': [{'secid': 'MRKZ', 'engine': 'stock', 'market': 'shares', 'board': 'EQNE'},              # 2008-09-04:2009-01-11 -листинг+дивы
                    {'secid': 'MRKZ', 'engine': 'stock', 'market': 'shares', 'board': 'EQNL'},              # 2009-01-12:2011-05-26
                    {'secid': 'MRKZ', 'engine': 'stock', 'market': 'shares', 'board': 'EQBS'},              # 2011-05-27:2013-08-30
                    {'secid': 'MRKZ', 'engine': 'stock', 'market': 'shares', 'board': 'TQBS'},              # 2013-09-02:2014-06-06
                    {'secid': 'MRKZ', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'}],             # 2014-06-09:наши дни
    'Детский Мир': [{'secid': 'DSKY', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'},              # 2017-02-10:2024-03-27 +листинг(85)+дивы
                    {'secid': 'DSKY', 'engine': 'stock', 'market': 'shares', 'board': 'TQPI'}],             # 2024-03-28:2024-10-14 +делистинг(71,5 до налога, 40-45 дней)
    'Полиметалл': [{'secid': 'POLY', 'engine': 'stock', 'market': 'shares', 'board': 'EQNL'},               # 2013-06-20:2013-08-05 -листинг+дивы
                   {'secid': 'POLY', 'engine': 'stock', 'market': 'shares', 'board': 'EQBR'},               # 2013-08-06:2013-08-30
                   {'secid': 'POLY', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'}],              # 2013-09-02:2024-09-23 +делистинг($3,5 после налога, сразу)
    'QIWI': [{'secid': 'QIWI', 'engine': 'stock', 'market': 'shares', 'board': 'TQNE'},                     # 2013-10-08:2014-06-05 -листинг+дивы
             {'secid': 'QIWI', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'}],                    # 2014-06-06:2025-11-17 +делистинг(210 после налога, 28 дней)
    'Трансаэро': [{'secid': 'TAER', 'engine': 'stock', 'market': 'shares', 'board': 'EQLV'},                # 2011-03-29:2011-08-17 -листинг-дивы
                  {'secid': 'TAER', 'engine': 'stock', 'market': 'shares', 'board': 'EQNL'},                # 2011-08-18:2013-08-30
                  {'secid': 'TAER', 'engine': 'stock', 'market': 'shares', 'board': 'TQNL'},                # 2013-09-02:2014-05-26
                  {'secid': 'TAER', 'engine': 'stock', 'market': 'shares', 'board': 'TQBS'},                # 2014-05-27:2014-06-06
                  {'secid': 'TAER', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'},                # 2014-06-09:2015-12-21
                  {'secid': 'TAER', 'engine': 'stock', 'market': 'shares', 'board': 'TQDE'}],               # 2015-12-22:2017-09-19 +банкротство
    'Уралкалий': [{'secid': 'URKA', 'engine': 'stock', 'market': 'shares', 'board': 'EQNE'},                # 2007-11-20:2010-09-08 -листинг+дивы
                  {'secid': 'URKA', 'engine': 'stock', 'market': 'shares', 'board': 'EQNL'},                # 2010-09-09:2011-06-07
                  {'secid': 'URKA', 'engine': 'stock', 'market': 'shares', 'board': 'EQBR'},                # 2011-06-08:2013-08-30
                  {'secid': 'URKA', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'}],               # 2013-03-25:2019-09-17 +делистинг(120 после налога, 25 дней)
    'Дикси': [{'secid': 'DIXY', 'engine': 'stock', 'market': 'shares', 'board': 'EQNE'},                    # 2007-06-26:2008-08-21 -листинг-дивы
              {'secid': 'DIXY', 'engine': 'stock', 'market': 'shares', 'board': 'EQNL'},                    # 2008-08-22:2013-04-10
              {'secid': 'DIXY', 'engine': 'stock', 'market': 'shares', 'board': 'EQBS'},                    # 2013-04-11:2013-08-30
              {'secid': 'DIXY', 'engine': 'stock', 'market': 'shares', 'board': 'TQBS'},                    # 2013-09-02:2013-11-05
              {'secid': 'DIXY', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'}],                   # 2013-11-06:2018-06-22 +делистинг(340 после налога, 15 дней)
    'РусГидро': [{'secid': 'HYDR', 'engine': 'stock', 'market': 'shares', 'board': 'EQLV'},                 # 2008-05-22:2008-06-02 -листинг+дивы
                 {'secid': 'HYDR', 'engine': 'stock', 'market': 'shares', 'board': 'EQNL'},                 # 2008-06-03:2008-08-21
                 {'secid': 'HYDR', 'engine': 'stock', 'market': 'shares', 'board': 'EQBR'},                 # 2008-08-22:2013-08-30
                 {'secid': 'HYDR', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'}],                # 2013-04-04:наши дни
    'Россети': [{'secid': 'FEES', 'engine': 'stock', 'market': 'shares', 'board': 'EQLV'},                  # 2008-07-16:2008-12-18 -листинг+дивы
                {'secid': 'FEES', 'engine': 'stock', 'market': 'shares', 'board': 'EQNL'},                  # 2008-12-19:2013-08-30
                {'secid': 'FEES', 'engine': 'stock', 'market': 'shares', 'board': 'TQNL'},                  # 2013-03-25:2014-06-02
                {'secid': 'FEES', 'engine': 'stock', 'market': 'shares', 'board': 'TQBS'},                  # 2014-06-03:2014-06-06
                {'secid': 'FEES', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'}],                 # 2014-06-09:наши дни
    'Селигдар': [{'secid': 'SELG', 'engine': 'stock', 'market': 'shares', 'board': 'EQNE'},                 # 2011-09-30:2011-12-05 -листинг+дивы
                 {'secid': 'SELG', 'engine': 'stock', 'market': 'shares', 'board': 'EQBR'},                 # 2011-12-06:2012-07-31
                 {'secid': 'SELG', 'engine': 'stock', 'market': 'shares', 'board': 'EQBS'},                 # 2012-08-01:2013-08-30
                 {'secid': 'SELG', 'engine': 'stock', 'market': 'shares', 'board': 'TQBS'},                 # 2013-09-02:2014-06-06
                 {'secid': 'SELG', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'}],                # 2014-06-09:наши дни
    'ВСМПО-АВИСМА': [{'secid': 'VSMO', 'engine': 'stock', 'market': 'shares', 'board': 'EQNL'},             # 2005-03-02:2011-12-05 -листинг+дивы
                     {'secid': 'VSMO', 'engine': 'stock', 'market': 'shares', 'board': 'TQNL'},             # 2013-09-02:2014-06-06
                     {'secid': 'VSMO', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'}],            # 2014-06-09:наши дни
    'Обувь России': [{'secid': 'OBUV', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'},             # 2017-10-20:2021-03-25 +листинг(140)+дивы
                     {'secid': 'ORUP', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'},             # 2021-03-26:2022-02-10
                     {'secid': 'ORUP', 'engine': 'stock', 'market': 'shares', 'board': 'TQPI'}],            # 2022-02-11:2023-05-16 +банкротство
    'Соликамск магн-завод': [{'secid': 'MGNZ', 'engine': 'stock', 'market': 'shares', 'board': 'EQNE'},     # 2011-02-17:2013-08-30 -листинг+дивы
                             {'secid': 'MGNZ', 'engine': 'stock', 'market': 'shares', 'board': 'TQNE'},     # 2013-09-04:2014-06-06
                             {'secid': 'MGNZ', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'}],    # 2014-06-09:2022-11-07 +делистинг(9 600 после налога, 900 дней)
    'ОВК': [{'secid': 'UWGN', 'engine': 'stock', 'market': 'shares', 'board': 'TQPI'},                      # 2021-12-10:2024-12-06 +листинг(700)-дивы
            {'secid': 'UWGN', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'}],                     # 2015-04-30:2021-12-09 + 2024-12-09:наши дни

    'ОФЗ, фикс 1+': [{'secid': 'RGBITR', 'engine': 'stock', 'market': 'index', 'board': 'SNDX'}],           # 2002-12-30:наши дни
    'ОФЗ, фикс 5-10': [{'secid': 'RUGBITR10Y', 'engine': 'stock', 'market': 'index', 'board': 'RTSI'}],     # 2010-12-30:наши дни
    'ВДО, фикс': [{'secid': 'RUCBITRB', 'engine': 'stock', 'market': 'index', 'board': 'RTSI'},             # 2010-12-30:2023-05-31
                  {'secid': 'RUCBHYTR', 'engine': 'stock', 'market': 'index', 'board': 'RTSI'}],            # 2021-03-26:наши дни
    #здесь могли быть линкеры, замещайки и ПК, недвижимость пока убрал(тк непонятно как ее правильно добавить)

    'Денежный рынок(LQDT)': [{'secid': 'LQDT', 'engine': 'stock', 'market': 'shares', 'board': 'TQTF'},     # 2022-07-22:2026-06-19
                             {'secid': 'LQDT', 'engine': 'stock', 'market': 'shares', 'board': 'TQBR'}],    # 2026-06-22:наши дни
    'Денежный рынок(REPO)': [{'secid': 'MOEXREPO', 'engine': 'stock', 'market': 'index', 'board': 'SNDX'}], # 2013-12-20:2024-12-30

    'Юань': [{'secid': 'CNYRUB_TOM', 'engine': 'currency', 'market': 'selt', 'board': 'CETS'}],             # 2013-04-15:наши дни
    'Доллар': [{'secid': 'USD000UTSTOM', 'engine': 'currency', 'market': 'selt', 'board': 'CETS'}],         # 2003-04-15:наши дни
    'Евро': [{'secid': 'EUR_RUB__TOM', 'engine': 'currency', 'market': 'selt', 'board': 'CETS'},            # 2005-06-20:2024-06-11
             {'secid': 'EURFIXME', 'engine': 'currency', 'market': 'index', 'board': 'FIXI'}],              # 2019-08-01:наши дни

    'Золото': [{'secid': 'GLDRUB_TOM', 'engine': 'currency', 'market': 'selt', 'board': 'CETS'}],           # 2013-10-21:наши дни
    'Серебро': [{'secid': 'SLVRUB_TOM', 'engine': 'currency', 'market': 'selt', 'board': 'CETS'}],          # 2014-01-10:наши дни
}

if __name__ == "__main__":
    storage = LocalCSVStorage()
    cleaner = DataCleaner(storage)
    delist_history = pd.read_csv('data/cache/delist_panel.csv').set_index('Company')

    logging.info("=== ЗАПУСК ГЛОБАЛЬНОЙ СИНХРОНИЗАЦИИ ДАННЫХ (ГОРИЗОНТ: С 2010 ГОДА) ===")

    returns_df, divs_df, volatility_panel = cleaner.build_cleaned_market_data(
        assets_config=assets_portfolio,
        target_start='2010-01-01',
        delist_history=delist_history
    )

    divs_matrix_renamed = divs_df.add_suffix('_div')

    global_board_panel = pd.concat([returns_df, divs_matrix_renamed], axis=1,sort=True)

    os.makedirs('data/matrix', exist_ok=True)

    global_board_panel.to_csv('data/matrix/global_board_panel.csv')
    volatility_panel.to_csv('data/matrix/global_volatility_panel.csv')

    logging.info("=== ГЛОБАЛЬНАЯ СИНХРОНИЗАЦИЯ УСПЕШНО ЗАВЕРШЕНА ===")
