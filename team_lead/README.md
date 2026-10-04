# Rolle: Team Lead

## Odpovědnosti

1. **Správa repozitáře**
   - Vytvářet a organizovat GitHub repozitář.
   - Nastavit strategii větví (např. `main` + feature větve).
   - Přidávat členy týmu jako spolupracovníky.

2. **Workflow Issues & PR**
   - Vytvářet issues pro každou podúlohu (data, model, pipeline, edge).
   - Recenzovat a slučovat pull requesty.
   - Používat štítky: `role:data-engineer`, `role:ai-ml` atd.

3. **Komunikace a koordinace**
   - Sleduovat pokrok v issues nebo project boardu.
   - Plánovat sync setkání mezi vlastníky rolí.

4. **Dokumentace**
   - Udržovat tento centralizovaný README.
   - Aktualizovat architekturní diagramy.
   - Dokumentovat rozhodynutí (např. YOLOv8 vs v11).

## Nastavení GitHubu

```bash
gh repo create AI-Detection-System --public
git push -u origin main
gh repo add-collaborator Hacktor1 <uživatelské_jméno>
```

## Šablona Project Board

Použij GitHub Project board se sloupci:
- `Backlog`
- `In Progress`
- `Review`
- `Done`

Označ položky podle rolí: `data-engineer`, `ai-ml`, `pipeline`, `edge`, `team-lead`.