"""Auditable mapping from all 51 original PDF pages to the reduced catalog."""
import csv,json
from pathlib import Path
from figure_catalog import FIGURES,MERGED,path_for
root=Path(__file__).parent
reference=json.loads((root/'review_reference_manifest.json').read_text())['pages']
rows=[]
for source in reference:
    number=source['page']
    if number in FIGURES:
        action='Пересчитан и сохранён';target=str(path_for(Path('results'),number));note=''
        if number==3:note='Три пары кривых: 50 и 200 устройств, три политики. Включает прежние №4-7.'
        if number==32:note='Ось X: угол запуска; кривые 1,16,64 повторений.'
        if number==20:note='Вид карты сохранён; данные пересчитаны с текущими 16 повторениями, шкала 0..1 не меняется.'
        if number in (2,8,18):note='Сетка числа устройств строго 20,40,...,200.'
    elif number in MERGED:
        action='Объединён';target=str(path_for(Path('results'),MERGED[number]));note='Отдельного построения больше нет.'
    else:
        action='Удалён из построения';target='';note='По замечанию научного руководителя.'
        if number in (37,38):note='Маргинальные события уже представлены в №33.'
        if number==39:note='Не строгий дубль общего отказа на попытку, но итоговый успех устройств уже есть в №2 и №3.'
    rows.append(dict(original_page=number,original_source=source['source'],action=action,new_source=target,note=note))
with (root/'review_changes.csv').open('w',newline='',encoding='utf-8-sig') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print('51 original pages mapped; 24 retained figures')
