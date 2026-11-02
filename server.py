from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid, ChartModule, TextElement
from mesa.visualization.UserParam import UserSettableParameter
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from recidivism_abm.model.recidivism_model import RecidivismModel


def filtered_portrayal(state_name):
    def inner(agent):
        if agent.justice_state != state_name:
            return None  # ⛔ Skip agents not in this state
        return agent_portrayal(agent)
    return inner

def agent_portrayal(agent):
    # ✅ Recidivated agents always appear black, regardless of justice state
    if agent.recidivated_agent == 1:
        color = "black"
    else:
            color = {
            "Free": "green",
            "Trial": "orange",
            "Prison": "red",
            "Supervision": "blue"
        }.get(agent.justice_state, "gray")

    return {
            "Shape": "circle",
            "Color": color,
            "Filled": "true",
            "Layer": 0,
            "r": 0.3
        }



# 🧭 Separate grids for each justice state
grid_trial = CanvasGrid(filtered_portrayal("Trial"), 40, 10, 800, 200)
grid_prison = CanvasGrid(filtered_portrayal("Prison"), 40, 10, 800, 200)
grid_supervision = CanvasGrid(filtered_portrayal("Supervision"), 40, 10, 800, 200)
grid_Free = CanvasGrid(filtered_portrayal("Free"), 40, 10, 800, 200)



# 📊 Chart module (optional)
chart_states = ChartModule(
    [  {"Label": "CalibrationError_3yr", "Color": "red"},
       {"Label": "CalibrationError_6yr", "Color": "orange"},
       {"Label": "CalibrationError_10yr", "Color": "purple"}
       ],
    data_collector_name="datacollector"
)

chart_justice_states = ChartModule(
    [
        {"Label": "Trial", "Color": "orange"},
        {"Label": "Prison", "Color": "red"},
        {"Label": "Supervision", "Color": "blue"},
        {"Label": "Free", "Color": "green"}
    ],
    data_collector_name='datacollector',
    canvas_height=250,
    canvas_width=600
)


# 📊 Chart = Recidivists by Year (optional)
chart_RY = ChartModule(
    [
       {"Label": "CumulativeRecidivismRate", "Color": "blue"},
       #  {"Label": "MonthlyRecidivismRate", "Color": "Green"}
       {"Label": "RecidivismRate_3yr", "Color": "Red"},
       {"Label": "RecidivismRate_6yr", "Color": "Orange"},
       {"Label": "RecidivismRate_9yr", "Color": "Purple"}
    ],
    data_collector_name='datacollector',
    canvas_height=250,
    canvas_width=600
)



# 🧠 Simulation metadata and agent counts
class SimulationInfo(TextElement):
    def render(self, model):
        trial = sum(1 for a in model.schedule.agents if a.justice_state == "Trial")
        prison = sum(1 for a in model.schedule.agents if a.justice_state == "Prison")
        supervision = sum(1 for a in model.schedule.agents if a.justice_state == "Supervision")
        free = sum(1 for a in model.schedule.agents if a.justice_state == "Free")
        recidivists = model.count_recidivists_during_study()
        total = len(model.schedule.agents)
        if model.current_month < model.warmup_months:
            total_study = 0
        else:
            total_study = sum(
                1 for a in model.schedule.agents
                if getattr(a, "study_eligible_agent", False)
            )


        warmup_end = model.warmup_months - 1
        study_start = model.warmup_months
        study_end = model.warmup_months + model.study_months - 1
        phase = "Warm-up" if model.current_month < model.warmup_months else "Study"

        # Progress calculation
        progress = round((model.current_month / model.max_months) * 100)
        progress_color = "#FFA500" if phase == "Warm-up" else "#4CAF50"


        return f"""
        <div style='font-size:13px; line-height:1.4'>
            <b>Simulation Month:</b> {model.current_month} &nbsp;&nbsp;
            <b>Phase:</b> <span style='color:{progress_color}'>{phase} Period</span><br>
            <b>Warm-up Period:</b> {model.warmup_months} months &nbsp;&nbsp <br>
            <b>Agents in Simulation ( Trial + Prison + Supervision + Free ):</b> {total}<br>
            
            <span style='color:orange'>Trial:</span> {trial} &nbsp;&nbsp;
            <span style='color:red'>Prison:</span> {prison} &nbsp;&nbsp;
            <span style='color:blue'>Supervision:</span> {supervision} &nbsp;&nbsp;
            <span style='color:green'>Free:</span> {free} &nbsp;&nbsp;<br>
            <span style='color:black'>Recidivist within 3 years:</span> {recidivists}<br><br>
            
            <b>Study Period:</b> {model.study_months} months<br>
            <b>Agents in Study Period:</b> {total_study}<br>
            <b>Progress:</b>
            <div style='width:100%; background-color:#eee; border-radius:5px; height:12px;'>
                <div style='width:{progress}%; background-color:{progress_color}; height:12px; border-radius:5px;'></div>
            </div>
            <span style='font-size:11px'>{progress}% complete</span>

            
        </div>
        """




# 🎨 Compact legend
class Legend(TextElement):
    def render(self, model):
        return """
                    <div style='
                        position: absolute;
                        top: 140px;
                        right: 50px;
                        font-size:12px;
                        background-color: white;
                        padding: 8px;
                        border: 1px solid #ccc;
                        border-radius: 5px;
                        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
                    '>
                        <b>Legend:</b><br>
                        <span style='color:orange'>● Trial</span><br>
                        <span style='color:red'>● Prison</span><br>
                        <span style='color:blue'>● Supervision</span><br>
                        <span style='color:green'>● Free</span><br>
                        <span style='color:black'>● Recidivists</span>
                    </div>

        """




class SliderValues(TextElement):
    def render(self, model):
        return f"""
        <div style='font-size:12px'>
        <b>Slider Settings:</b><br>
        Initial Agents: {model.initial_agents}<br>
        Warm-up Months: {model.warmup_months}<br>
        Study Months: {model.study_months}<br>
        Monthly Intake: {model.monthly_intake}<br>
        #Target Recidivism: {model.target_recidivism}
        </div>
        """


# 🎛 Slider-controlled parameters
model_params = {
    "initial_agents": UserSettableParameter("slider", "Initial Agents", 1000, 1000, 100000, 500),
    "warmup_months": UserSettableParameter("slider", "Warm-up Period (months)", 144,0, 240, 12),
    "study_months": UserSettableParameter("slider", "Study Period (months)", 108, 0, 120, 12),
    "monthly_intake": UserSettableParameter("slider", "Monthly Intake (% of Initial Agents during Warmup)", 10, 1, 100, 1),
 
    "bias_factor": UserSettableParameter("slider", "Bias Factor (%)", 0.03,0.03, 0.01, 0.2),
    # ✅ Add checkbox for peer influence
    "enable_peer_influence": UserSettableParameter("checkbox", "Enable Recidivist Cellmate Influence", True)
    #"target_recidivism": UserSettableParameter("slider", "Target Recidivism Rate", 0.68, 0.0, 1.0, 0.01)
}

# 🚀 Launch server
server = ModularServer(
    RecidivismModel,
    [
        SimulationInfo(),
        Legend(),
        chart_justice_states,
        grid_trial,
        grid_prison,
        grid_supervision,
        grid_Free,
        chart_states,
        chart_RY
    ],
    "Recidivism ABM (NIJ Schema)",
    model_params
)
