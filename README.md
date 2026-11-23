# Near-Earth Object Hazard Classification — Random Forest vs XGBoost

Predicts whether an asteroid is **potentially hazardous** using real NASA data, and compares
Random Forest (bagging) with XGBoost (boosting).

* Project code: [neo_hazard_python/](neo_hazard_python/) (setup in its `README.md`, findings in `REPORT.md`)
* Original brief: [basic.md](basic.md)
* **Presentation prep / viva notes: [prepare.md](prepare.md)**

```powershell
cd neo_hazard_python
.\.venv\Scripts\activate
python run_analysis.py      # full study, writes results/
streamlit run app.py        # dashboard at http://localhost:8501
pytest                      # 32 tests
```
