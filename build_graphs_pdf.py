"""One current v16 figure and its actual input parameters per PDF page."""
import json,re,hashlib
from pathlib import Path
from xml.sax.saxutils import escape
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A2,A3,landscape
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Table,TableStyle,Paragraph
from reportlab.lib.styles import ParagraphStyle
ROOT=Path(__file__).parent
PROJECT=ROOT;RESULTS=PROJECT/'results';OUT=ROOT/'report'
OUT.mkdir(parents=True,exist_ok=True)
pdfmetrics.registerFont(TTFont('DejaVu',str(ROOT/'assets/fonts/DejaVuSans.ttf')))
pdfmetrics.registerFont(TTFont('DejaVuBold',str(ROOT/'assets/fonts/DejaVuSans-Bold.ttf')))
STYLE=ParagraphStyle('cell',fontName='DejaVu',fontSize=10,leading=13,textColor=colors.HexColor('#172B3A'))
SECTIONS={'core':'Основные результаты','mapping':'Отображение ресурсов','repetitions':'Повторения преамбулы','trigger':'Порог общего запуска','phy':'Чувствительность детектора и нагрузки','power':'Электрические мощности','channel_map':'Карта индивидуальных каналов','threshold_wait':'Ожидание индивидуального порога'}
TITLES={'01_failure_composition':'Причины отказов при разных углах запуска','02_access_success':'Доля устройств с успешным доступом','07_energy_per_success':'Энергия на успешный доступ','10_energy_reliability':'Энергия и надёжность доступа','energy_unsuccessful':'Энергия неуспешных устройств','success_ratio':'Доля успешных устройств','preamble_collision_probability':'Вероятность коллизии преамбулы','payload_collision_probability':'Вероятность коллизии полезной нагрузки','delay_s_mean':'Средняя задержка до успеха','delay_s_p95':'95-й процентиль задержки до успеха','energy_per_success_j':'Энергия на успешный доступ','01_failure_vs_trigger_elevation':'Маргинальные события отказа при разных углах запуска','03_capped_wait':'Среднее ожидание порога с ограничением горизонтом','individual_channels_100':'Индивидуальный BLER блока данных: 100 устройств'}
STATS={'mean':'среднее','median':'медиана','p90':'90-й процентиль','p95':'95-й процентиль'}

def fmt(v):
    if isinstance(v,(int,float)):return f'{v:g}'.replace('.',',')
    return str(v)
def values(cs,key,unit='',default='Не применяется'):
    vs=[]
    for c in cs:
        v=c.get(key)
        if v is None and key.endswith('_mw') and key[:-3]+'_w' in c:v=c[key[:-3]+'_w']*1000
        if v is not None:
            vs.extend(v if isinstance(v,list) else [v])
    unique=list(dict.fromkeys(vs))
    if not unique:return default
    if all(isinstance(v,(int,float)) for v in unique):unique=sorted(unique)
    return '; '.join(fmt(v) for v in unique)+((' '+unit) if unit else '')
def title(p):
    stem=p.stem
    if p.parent.parent.name=='repetitions' and stem=='success_ratio':return 'Успех от угла запуска: 1, 16 и 64 повторения'
    if stem=='03_completion_combined':return 'Накопленный успех: три пары кривых для 50 и 200 устройств'
    if stem.startswith('03_completion_'):return 'Накопленная доля успешных устройств: N='+str(int(float(stem.split('_')[-1])))
    for prefix,label in [('04_delay_','Задержка до успешного доступа'),('08_energy_until_success_','Энергия до первого успеха')]:
        if stem.startswith(prefix):return label+': '+STATS[stem[len(prefix):]]
    return TITLES.get(stem,stem)

def parameters(p):
    suite=p.relative_to(RESULTS).parts[0];stem=p.stem;note=''
    if suite=='repetitions' and stem=='success_ratio':
        settings=json.loads((RESULTS/'repetition_elevation/settings.json').read_text());cs=[settings['config']|{'repetitions':nr} for nr in (1,16,64)];n='50';passes=values(cs,'passes');threshold='Угол запуска 10-60°; эталон 16 повторений';note='Три кривые 1/16/64 повторений при одинаковом угле начала доступа. Не мгновенный BLER.'
    elif stem=='01_failure_composition':
        settings=json.loads((RESULTS/'failure_curves/settings.json').read_text());cs=[settings['config']];n='50; условная плотность 50 устройств/км²';passes=values(cs,'passes');threshold='Угол запуска 10-60°, шаг 5°';note='Взаимоисключающие причины и их сумма на выполненную попытку за оставшееся окно пролёта. Не мгновенная BLER.'
    elif suite=='channel_map':
        settings=json.loads((p.parent/'report.json').read_text());cs=[settings['settings']];n='100 на каждую среду';passes='Один случайный снимок';threshold='Угол снимка '+fmt(settings['elevation_deg'])+'°';note='Эмпирическая приёмная кривая блока данных; коллизии не учитываются. Фон зданий не определяет затенение.'
    elif suite=='threshold_wait':
        settings=json.loads((p.parent/'settings.json').read_text());cs=[settings['config']];n=values(cs,'sensors');passes=values(cs,'passes');threshold='Индивидуальный SNR от -5 до 15 дБ, шаг 1 дБ';note='Ожидание от восходящих 10°; не достигшие порога включены с ограничением горизонтом. Не общий триггер RA.'
    else:
        cs=list(json.loads((RESULTS/suite/'configs_used.json').read_text()).values());passes=values(cs,'passes');threshold=values(cs,'access_trigger_snr_db','дБ')
        n=values(cs,'sensor_counts') if suite=='core' else '50 на рисунке'
        if stem=='03_completion_combined':n='50 и 200; три пары кривых'
        elif stem.startswith('03_completion_'):n=str(int(float(stem.split('_')[-1])))
        elif suite=='core':n='20,40,...,200'
        if suite=='repetitions':note='Общий старт фиксирован по эталонным 128 повторениям; меняются фактические повторения и длительность передачи.'
        elif suite=='phy':note='Две однофакторные серии: порог SINR нагрузки и P_FA. Не декартово произведение настроек.'
        elif suite=='power':note='Однофакторные серии мощностей; RX выключен (MsgB отсутствует). Изменение одной RX-мощности не меняет энергию.'
        elif suite=='trigger':note='Порог запуска преобразуется в фактический угол. Низкие пороги могут соответствовать одному углу 10°.'
        else:note='Успех на устройство и успех на попытку различаются; задержки и энергия до успеха относятся только к успешным.'
    policies='Одна попытка; 5 с; U[1;5] с'
    if suite in ('channel_map','threshold_wait'):policies='Не применяется к этой диагностике'
    data=[('Устройства на рисунке',n),('Независимые пролёты',passes),('Среды','Город; деревня; сельская с растительностью'),('Seed',values(cs,'seed')),
          ('Порог / условие запуска',threshold),('Политики повторов',policies),('Повторения преамбулы',values(cs,'repetitions')),('Повторения нагрузки',values(cs,'pusch_repetitions')),
          ('Пул преамбул',values(cs,'preamble_pool_size')),('Преамбул на ресурс нагрузки',values(cs,'preamble_to_pusch_ratio')),
          ('Период RA / смещение',values(cs,'ra_period_ms','мс')+' / '+values(cs,'ra_offset_ms','мс')),('Порог SINR нагрузки / P_FA',values(cs,'pusch_sinr_threshold_db','дБ')+' / '+values(cs,'false_alarm_probability')),
          ('Высота / максимум угла',values(cs,'altitude_km','км')+' / '+values(cs,'max_elevation_deg','°')),('Частота / EIRP датчика',values([{'ghz':c['frequency_hz']/1e9} for c in cs],'ghz','ГГц')+' / '+values(cs,'uplink_tx_eirp_dbm','дБм')),
          ('Усиление RX / потери',values(cs,'satellite_rx_gain_dbi','дБи')+' / '+values(cs,'system_loss_db','дБ')),('Шум: T / B / NF',values(cs,'temperature_k','K')+' / '+values([{'khz':c['bandwidth_hz']/1000} for c in cs],'khz','кГц')+' / '+values(cs,'noise_figure_db','дБ')),
          ('TX / RX, электрические',values(cs,'power_preamble_tx_mw','мВт')+' / '+values(cs,'power_rx_mw','мВт')),('Idle / сон',values(cs,'power_idle_mw','мВт')+' / '+values(cs,'power_sleep_mw','мВт')),
          ('Преамбула: разрыв',values(cs,'preamble_gap_ms','мс')),('Нагрузка: задержка / длительность',values(cs,'payload_offset_ms','мс')+' / '+values(cs,'payload_duration_ms','мс'))]
    if suite in ('channel_map','threshold_wait'):
        data=[(k,('Не применяется' if k in ('Пул преамбул','Преамбул на ресурс нагрузки','Порог SINR нагрузки / P_FA','Повторения нагрузки','Нагрузка: задержка / длительность') else v)) for k,v in data]
    return suite,data,note

from figure_catalog import FIGURES,path_for
numbers=sorted(FIGURES);paths=[path_for(RESULTS,n) for n in numbers]
assert all(p.exists() for p in paths),'Run the full simulation before building the PDF'
assert set(paths)==set(RESULTS.rglob('*.png')),'Unexpected or removed graphs in results'
target=OUT/'sensor_model_v17_reviewed_graphs.pdf';doc=canvas.Canvas(str(target),pageCompression=1)
doc.setTitle('LEO NB-IoT: графики после замечаний научного руководителя, v17');doc.setAuthor('Исследовательская модель LEO NB-IoT')
manifest=[]
for page,p in enumerate(paths,1):
    suite,data,note=parameters(p);iw,ih=Image.open(p).size
    width,height=landscape(A2 if iw/ih<2 else A3);doc.setPageSize((width,height));margin=32;usable=width-2*margin
    doc.setFillColor(colors.HexColor('#16465D'));doc.setFont('DejaVuBold',17);doc.drawString(margin,height-31,f'Исходный №{numbers[page-1]}. '+title(p))
    doc.setFillColor(colors.HexColor('#536B78'));doc.setFont('DejaVu',10)
    doc.drawString(margin,height-49,'LEO NB-IoT | v17 | '+SECTIONS[suite]);doc.drawRightString(width-margin,height-49,f'{page} / {len(paths)}')
    cells=[[Paragraph('<b>Параметр</b>',STYLE),Paragraph('<b>Значение</b>',STYLE),Paragraph('<b>Параметр</b>',STYLE),Paragraph('<b>Значение</b>',STYLE)]]
    for i in range(0,len(data),2):
        pair=data[i:i+2];cells.append([Paragraph(escape(str(s)),STYLE) for item in pair for s in item])
    table=Table(cells,colWidths=[usable*.18,usable*.32,usable*.18,usable*.32])
    table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E3EEF3')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F4F7F9')]),('VALIGN',(0,0),(-1,-1),'TOP'),('BOX',(0,0),(-1,-1),.5,colors.HexColor('#C4D3DC')),('INNERGRID',(0,0),(-1,-1),.3,colors.HexColor('#D6E0E6')),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
    _,th=table.wrap(usable,height);table_y=57;table.drawOn(doc,margin,table_y)
    top=height-62;bottom=table_y+th+12;available=top-bottom
    scale=min(usable/iw,available/ih);dw,dh=iw*scale,ih*scale
    assert available>200,(p,available)
    doc.drawImage(str(p),margin+(usable-dw)/2,bottom+(available-dh)/2,dw,dh,mask='auto')
    para=Paragraph(escape(note),ParagraphStyle('note',parent=STYLE,fontSize=9,leading=11))
    _,nh=para.wrap(usable,30);assert nh<=30,(p,nh);para.drawOn(doc,margin,table_y-7-nh)
    doc.setFont('DejaVu',8);doc.setFillColor(colors.HexColor('#617783'))
    doc.drawString(margin,14,'Исследовательские допущения; без MsgB и capture. Мощность - мВт, энергия - мДж. Источник: '+str(p.relative_to(PROJECT)))
    doc.bookmarkPage(f'p{page}');doc.addOutlineEntry(title(p),f'p{page}',0)
    manifest.append(dict(page=page,original_page=numbers[page-1],source=str(p.relative_to(PROJECT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),page_size=[width,height]))
    doc.showPage()
doc.save()
(OUT/'pdf_manifest.json').write_text(json.dumps(dict(pages=manifest),ensure_ascii=False,indent=2))
print(target);print('Pages:',len(paths))
