# Team Lead Role

## Responsibilities

1. **Repository Management**
   - Create and organize the GitHub repository.
   - Set up branch strategy (e.g., `main` + feature branches).
   - Add team members as collaborators.

2. **Issue & PR Workflow**
   - Create issues for each subtask (data, model, pipeline, edge).
   - Review and merge pull requests.
   - Use labels: `role:data-engineer`, `role:ai-ml`, etc.

3. **Communication & Coordination**
   - Track progress in issues or a project board.
   - Schedule sync-ups between role owners.

4. **Documentation**
   - Maintain this centralized README.
   - Keep architecture diagrams updated.
   - Document decisions (e.g., YOLOv8 vs v11).

## GitHub Setup

```bash
gh repo create AI-Detection-System --public
git push -u origin main
gh repo add-collaborator Hacktor1 <username>
```

## Project Board Template

Use a GitHub Project board with columns:
- `Backlog`
- `In Progress`
- `Review`
- `Done`

Tag items per role: `data-engineer`, `ai-ml`, `pipeline`, `edge`, `team-lead`.