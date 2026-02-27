import streamlit as st
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from datetime import datetime, timedelta
import os

st.set_page_config(page_title="AI Trip Planner", page_icon="✈️", layout="wide")

st.session_state.setdefault("trip_plan", None)

def create_agents(api_key: str):
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)

    city_expert = Agent(
        role="City Information Expert",
        goal="Provide comprehensive information about cities including attractions, culture, and local tips",
        backstory="Experienced travel researcher with deep knowledge of cities worldwide.",
        verbose=True,
        allow_delegation=False,
        llm=llm
    )
    itinerary_planner = Agent(
        role="Itinerary Planner",
        goal="Create detailed, personalized day-by-day travel itineraries based on user preferences",
        backstory="Professional travel planner with years of experience creating customized trips.",
        verbose=True,
        allow_delegation=False,
        llm=llm
    )
    return city_expert, itinerary_planner

def create_tasks(city_expert, itinerary_planner, origin, destination, start_date, end_date, interests, budget, travel_style):
    duration = (end_date - start_date).days + 1
    interests_str = ", ".join(interests)

    research_task = Task(
        description=f"""Research {destination} and provide:
- Top 5 must-visit attractions related to: {interests_str}
- Best restaurants and local cuisine
- Cultural highlights and local customs
- Transportation options within the city
- Weather considerations and safety tips
Focus on {budget} budget and {travel_style} travel style. Be concise and practical.""",
        agent=city_expert,
        expected_output="Structured city info with practical travel tips"
    )

    itinerary_task = Task(
        description=f"""Create a {duration}-day itinerary from {origin} to {destination}.
Dates: {start_date.strftime('%B %d, %Y')} to {end_date.strftime('%B %d, %Y')}
Interests: {interests_str} | Budget: {budget} | Style: {travel_style}

For each day include:
- Morning (9 AM-12 PM), Afternoon (12-6 PM), Evening (6-10 PM) with specific locations
- Estimated daily cost range
- Transportation tips""",
        agent=itinerary_planner,
        expected_output="Day-by-day itinerary with activities, timings, and practical details",
        context=[research_task]
    )

    return [research_task, itinerary_task]

def generate_trip_plan(api_key, origin, destination, start_date, end_date, interests, budget, travel_style):
    try:
        city_expert, itinerary_planner = create_agents(api_key)
        tasks = create_tasks(city_expert, itinerary_planner, origin, destination, start_date, end_date, interests, budget, travel_style)
        result = Crew(agents=[city_expert, itinerary_planner], tasks=tasks, verbose=True).kickoff()
        return str(result)
    except Exception as e:
        return f"Error generating trip plan: {e}\n\nPlease check your OpenAI API key and try again."

# Sidebar
with st.sidebar:
    st.header("Configuration")
    api_key = os.getenv("OPENAI_API_KEY", "") or st.text_input("OpenAI API Key (required)", type="password")

# Main UI
st.title("AI-Powered Trip Planner")
st.markdown("### Plan Your Perfect Trip with CrewAI & OpenAI")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Trip Details")
    origin = st.text_input("Origin City", value="New York")
    destination = st.text_input("Destination City", value="Paris")
    start_date = st.date_input("Start Date", value=datetime.now() + timedelta(days=7), min_value=datetime.now())
    end_date = st.date_input("End Date", value=datetime.now() + timedelta(days=12), min_value=start_date)
    st.info(f"Trip Duration: {(end_date - start_date).days + 1} days")

with col2:
    st.subheader("Preferences")
    interests = st.multiselect(
        "Your Interests",
        ["Culture & Museums", "Food & Dining", "Adventure & Sports", "Nature & Parks",
         "Shopping", "Nightlife", "History", "Beach & Water Activities", "Art & Architecture"],
        default=["Culture & Museums", "Food & Dining"]
    )
    budget = st.select_slider("Budget Range", options=["Budget", "Moderate", "Comfortable", "Luxury"], value="Moderate")
    travel_style = st.radio("Travel Style", ["Relaxed", "Balanced", "Packed"], index=1, horizontal=True)

st.markdown("---")
_, col_btn, _ = st.columns([1, 2, 1])

with col_btn:
    if st.button("Generate AI Trip Plan", type="primary", use_container_width=True):
        if not api_key:
            st.error("Please enter your OpenAI API key in the sidebar.")
        elif not origin or not destination:
            st.error("Please enter both origin and destination cities.")
        elif not interests:
            st.error("Please select at least one interest.")
        else:
            with st.status("AI Agents are planning your trip...", expanded=True) as status:
                st.write("City Expert is researching your destination...")
                st.write("Itinerary Planner is creating your schedule...")
                st.session_state.trip_plan = generate_trip_plan(
                    api_key, origin, destination, start_date, end_date, interests, budget, travel_style
                )
                status.update(label="Trip plan generated!", state="complete", expanded=False)

if st.session_state.trip_plan:
    duration = (end_date - start_date).days + 1
    st.success("Your personalized trip plan is ready!")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Route", f"{origin} to {destination}")
    c2.metric("Duration", f"{duration} days")
    c3.metric("Budget", budget)
    c4.metric("Style", travel_style)

    st.info(f"Interests: {', '.join(interests)} | Dates: {start_date.strftime('%b %d')} - {end_date.strftime('%b %d, %Y')}")
    st.markdown("### Your Personalized Trip Plan")
    st.markdown(st.session_state.trip_plan)

    col_dl, col_new, _ = st.columns([1, 1, 2])
    col_dl.download_button(
        "Download Plan",
        data=st.session_state.trip_plan,
        file_name=f"trip_{destination.lower().replace(' ', '_')}_{start_date.strftime('%Y%m%d')}.txt",
        mime="text/plain",
        use_container_width=True
    )
    if col_new.button("New Plan", use_container_width=True):
        st.session_state.trip_plan = None
        st.rerun()

st.markdown("---")
st.markdown("<div style='text-align:center;color:gray'>Built with Streamlit & CrewAI | Powered by OpenAI</div>", unsafe_allow_html=True)

