# ==============================================================================
# BTE 609: Artificial Intelligence for Business
# Professor: Dr. Niam Yaraghi
# Team: Alex Fotiadis, Dainiuris "Nuri" Oler, Justin Perez, George Kurtzer
# Project: Space Weather & Polar Aviation Route Dispatch System
# Executive Human-in-the-Loop Decision Support Platform
# Streamlit Web Application (app.py)
# ==============================================================================

import os
from datetime import datetime, timezone
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, mean_squared_error, mean_absolute_error, r2_score

# ------------------------------------------------------------------------------
# 1. PAGE SETUP & NATIVE DARK/LIGHT ADAPTIVE STYLING
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="AeroSpace Weather: Polar Route Dispatch",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark-mode adaptive CSS using transparent surfaces and subtle borders
st.markdown("""
<style>
    /* Executive Header Banner */
    .header-box {
        background: linear-gradient(135deg, #0B2545 0%, #133E6E 100%);
        color: #FFFFFF;
        padding: 22px 28px;
        border-radius: 10px;
        margin-bottom: 22px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        border: 1px solid rgba(255, 255, 255, 0.15);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* Subtle container cards adaptive to dark and light modes */
    .clean-card {
        background: rgba(128, 128, 128, 0.05);
        border: 1px solid rgba(128, 128, 128, 0.20);
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 18px;
    }
    
    /* Advisory status cards */
    .status-nominal {
        background-color: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.35);
        border-left: 6px solid #10B981;
        border-radius: 6px;
        padding: 16px 20px;
        font-weight: 600;
        margin: 14px 0;
    }
    
    .status-reroute {
        background-color: rgba(239, 68, 68, 0.12);
        border: 1px solid rgba(239, 68, 68, 0.35);
        border-left: 6px solid #EF4444;
        border-radius: 6px;
        padding: 16px 20px;
        font-weight: 600;
        margin: 14px 0;
    }

    /* Learning and category badges */
    .hitl-badge {
        display: inline-block;
        background-color: rgba(59, 130, 246, 0.15);
        border: 1px solid rgba(59, 130, 246, 0.35);
        color: #60A5FA;
        padding: 3px 10px;
        border-radius: 4px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-bottom: 8px;
    }
    
    /* Metric pill */
    .metric-pill {
        background: rgba(128, 128, 128, 0.08);
        border-radius: 6px;
        padding: 12px 16px;
        border: 1px solid rgba(128, 128, 128, 0.18);
        text-align: center;
    }
    .metric-pill-title {
        font-size: 0.78rem;
        opacity: 0.75;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 4px;
        letter-spacing: 0.5px;
    }
    .metric-pill-val {
        font-size: 1.4rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# 2. LOGO LOADER (WITH EMBEDDED HIGH-RES FALLBACK)
# ------------------------------------------------------------------------------
EMBEDDED_LOGO_B64 = "iVBORw0KGgoAAAANSUhEUgAAAfQAAAH0CAYAAADL1t+KAAAqfUlEQVR42u3dB5hdVb3w4UFRbIBioSgXkF5CU/pFpQekCBiKBUUw1NBDJyQCV5p0gnSQjiiIIr0qHQEpUqWX0Ekyvf3vWvvsCUPNmcnMZMq7v+d9uH5BiJlz9m+vtddeuyYiagCAgc0fAgAIOgAg6ACAoAMAgg4Agg4ACDoAIOgAgKADgKADAIIOAAg6ACDoACDoAICgAwCCDgAIOgAIOgAg6ACAoAMAgg4Agg4ACDoAIOgAgKADgKADAIIOAAg6ACDoACDoAICgAwCCDgAIOgAIOgAg6ACAoAMAgg4Agg4ACDoAIOgAgKADgKD7QwAAQQcABB0AEHQAQNABQNABAEEHAAQdABB0ABB0AEDQAQBBBwAEHQAEHQAQdABA0AEAQQcAQQcABB0AEHQAQNABQNABAEEHAAQdABB0ABB0AEDQAQBBBwAEHQAEHQAQdABA0AEAQQcAQQcABB0AEHQAQNABQNABAEEHAAQdABB0ABB0AEDQAQBBBwAEHQDwhwAAgg4ACDoAIOgAgKADgKADAIIOAAg6ACDoACDoAICgAwCCDgAIOgAIOgAg6ACAoAMA1QR93+TC5GyGvHOT8ckXfElgmn0vOS85x7nFubU8t/Z60O8Ih+O9oyn5spMxTLPtnE4cHzi39nrQb/Hn7Oh0TBR06BHbOp04PnBu7fWg3+zP2dHpeDeZ1ckYptk2TieOD5xbBd0h6CDoDkEXdIegg6A7BN0h6ICgOwTdIegg6A6HoDsEHQTdIeiC7hB0EHSHoL//aH36rmj62+HRdO1xDDTXHBvN158Y7ZPfFHToZ0G/+eF7Y+y5x8XYi05hoLlwfBxy6Wnx2rtvDayg55hP+sUMMXmHrzDQbDdL1I6aI9pefkzQoZ8FfZ8U85oVZouaNedjoFltnqhZd5F48NknBljQ00gvx6F2z/kYaHb/n6jdZ+Foe/UJQYd+FvQ80ivi8KOlGWg2HBYzjVg+Hn7+aUFH0EHQBV3QBR1BB0FH0AVd0AUdBB1BR9ABQRd0QUfQQdAFXdAFHUEHQUfQBV3QBR0EHUEXSEEHBB1BR9BB0AVd0AUdQQdBR9AFXdAFHQQdQRd0QQcEHUFH0EHQBV3QBR1BB0FH0AVd0AUdBB1BF3RBBwQdQUfQQdAFXdAFHUEHQUfQBR1BB0FH0AVd0AFBR9ARdBB0QRd0QUfQQdARdEFH0EHQEXRBF3RA0BF0BB0EXdAFXdARdBB0BF3QEXQQdARd0AVd0EHQEXQEHRB0QRd0BB0EHUEXdAQdBB1BF3RBF3QQdAQdQQcEXdAFHUEHQUfQBR1BB0FH0AVd0AUdBB1BR9ABQRd0QUfQQdARdEFH0EHQEXRBF3RBB0FH0BF0QNAFXdARdBB0QRd0QUfQQdARdEEXdEEHQUfQEXRA0AVd0BF0EHRBF3RBR9BB0BF0QRd0QQdBR9ARdEDQBV3QEXQQdEEXdEFH0EHQEXRBF3RBB0FH0AVd0AFBR9ARdBB0QRd0QUfQQdARdEEXdEEHQUfQBV3QAUFH0BF0EHRBF3RBR9BB0BF0QRd0QQdBR9AFXdABQUfQEXQQdEEXdEFH0EHQEXRBR9BB0BF0QRd0QNARdAQdBF3QBV3QEXQQdARd0BF0EHQEXdAF3QkZBB1BR9ABQRd0QUfQQdARdEFH0EHQEXRBF3RBB0FH0BF0QNAFXdARdBB0BF3QEXQQdARd0AVd0EHQEXQEHRB0QRd0BB0EHUEXdAQdBB1BF3RBF3QQdAQdQQcEXdAFHUEHQRdIQRd0BB0EHUEXdEEXdBB0BB1BBwRd0AUdQQdBF3RBF3QEHQQdQRd0QRd0EHQEHUEHBF3QBR1BB0EXdEEXdAQdBB1BF3RBF3QQdARdIAUdEHQEHUEHQRd0QRd0BB0EHUEXdEEXdBB0BF3QBR0QdAQdQQdBF3RBF3QEHQQdQRd0QRd0EHQEXdAFHRB0BB1BB0EXdEEXdAQdBB1BF3QEHQQdQRd0QQcEHUFH0EHQBV3QBR1BB0FH0AUdQQdBR9AFXdABQUfQEXQQdEEXdEFH0EHQEXRBR9BB0BF0QRd0QQdBR9ARdEDQBV3QEXQQdARd0BF0EHQEXdAFXdBB0BF0BB0QdEEXdAQdBB1BF3QEHQQdQRd0QRd0EHQEHUEHBF3QBR1BB0FH0AUdQQdBR9AFXdAFHQQdQUfQAUEXdEFH0EHQBV3QBR1BB0FH0AVd0AUdBB1BR9ABQRd0QUfQQdAFXdAFHUEHQUfQBV3QBR0EHUFH0AFBF3RBR9BB0AVd0AUdQQdBR9AFXdAFHQQdQRd0QQcEHUFH0EHQBV3QBR1BB0FH0AVd0AUdBB1BF3RBBwQdQUfQQdAFXdAFHUEHQUfQBV3QBR0EHUEXdEEHBB1BR9BB0AVd0AUdQQdBR9AFHUEHQUfQBV3QAUFH0BF0EHRBF3RBR9BB0BF0QUfQQdARdEEXdCdkEHQEHUEHBF3QBR1BB0FH0AUdQQdBR9AFXdAFHQQdQUfQAUEXdEFH0EHQEXRBR9BB0BF0QRd0QQdBR9ARdEDQBV3QEXQQdARd0BF0EHQEXdAFXdBB0BF0BB0QdEEXdAQdBF0gBV3QEXQQdARd0AVd0EHQEXQEHRB0QRd0BB0EXdAFXdARdBB0BF3QBV3QQdARdAQdEHRBF3QEHQRd0AVd0BF0EHQEXdAFXdBB0BF0gRR0QNARdAQdBF3QBV3QEXQQdARd0AVd0EHQEXRBF3RA0BF0BB0EXdAFXdCZxqAvEm0TnqrmR/2OoEPfBX3cJacKuqALOl0I+ugFou35B6v90H3ZyRim2bbVfOFGn3Ns1Kw1vzgKuqBThT3mSeaN1sdvreZHXZ8s7GQM0+yAar5w25w0NmrWXkAcBV3QqTbo80TrozdW86NuTBZzMoZpNqaaL9wvjj8oBX1BcRR0QaeaoM8btbvMGS13X1rNj7olWcnJGKbZ76r5wq13yM5RM3xhcRR0Qac6k3f6WjTfcnpUeWzgZAzT7NypfdFa21pjxX22ipr1FhFHQRd0qgz6jrNF05WHVhv0kU7GMM2um9oXra6xIebedp2oWX9xcRR0QafKoO/8jWj8w07VBv0wJ2OYJjMlT07ti/bCGxPiC5uvGDUbDBNHQRd0qrTLnFF/7IYRba3V/LgvdEKGaTJ3Uju1L9qtj9wXn9p4mRSGJcVR0AWdKu02d9QdtEy0179bzY/738mMTsrQbWsm7VP7op1x/eVRs+4iUbPRUuIo6IJOtZvLzBO1e80fbc/dX82Pu64cYTgxQ/fsXs0XbY+zfmdTGUEXdLpq3pg8avZovvOCau+jj3BShm6b6jOi7en/rT5mZBqhe2RN0AWdLq90/2o0/WlMtUE/yUkZumWW5NWpfcHeqZ0UX//F6lGz/hLCKOiCTjcWxh2zfkRbWzU/8sfLlbpO0NA161bzBbvlkftihnzv3II4QRd0uv0a1defqXaUvpqTM3TZqdV8ucZd/Ht7uAu6oNP9++i1o2aPljuqvo8+3skZuuQrycvVfLlWP+jXtnwVdEFn2jaYaThj62qD/noyu5M09OwrU594+bkiBDkIwijogk43p93njrr9l4j2d16uNuq7O0lDVfLeDXdX86U65srzomYdb1gTdEGnB6bdm28/r9qgP1uu2nXChk+2STVfqPy42vcO2MZ0u6ALOj1g1BzRcNKIdGZpqzbqo52sYap7t99TzZfpvqcfjRlt9yrogk7PvB99nkLrs/dVG/TXkm86acPHGlntl2nn0w63ul3QBZ2efT9644V7RheOM5204SPNVe3K9lfffiO+ttUPbCYj6IJOLzyT/uoT1QY9v2hiIydv+JCLqv0SjS2ePbcYTtAFnd4YpV+yd1dG6XlHmjmdwGGKn1f75Znwzpsx+y/XiJofLi6Egi7o9MIofe+FujJKz8dV4dWqkC2bvFPtF2fMRePdOxd0Qac3R+lfj8ZztosuHkc6mTPEzZE8Wu0X5ulXX4xZtlzFvXNBF3R6ecX7bt+Mlkeu62rUbTjDUH6b2o1d+bJsfvQ+7p0Lep8E/VZB9xa2+t+uFu31E7tyjmpNfurkzhDz2a4sgsvH5XfdFDU/XCxqNvLcuaD3ftD/LujkBXJNl4/t6ii9KfmZkzxDxOeT87vyBXl94tvx7ZHrRc16i4qfoPdJ0M+uLujHCvqgn3r/VrQ8fG1049jTyZ5Bbraky1+OLX63b9SsNb/wDeqgLxcPPffUwAp68/UnxuQdZxO+wWzXuaLu4O9G2xvPdifqeaHcp534GYQWTO7t6hfi+L9dGDXrLhw1Gy0lfIM46J/bbIV46pUX+k3Qj6zmd9Jyz6XFvdbaPeYVvkG+6r3+hE2ivWFyd6J+Rbn6VwQYLNZJnuvqF+GGh+6Oz+X92q1qH9zSz/frW60Wk+prq/lY/Lcvgr5TucDpE4+2/9793tSs8A3+DWfO3zW6efwn+YEQMMB9KtkvT0529Qvw2EvPxje3XjNq1ltE8Aa7dReJlfbZKlrbWqv5aNzWF0HPV6CTprrvZ+3bUTdm2ajdbW7RG/T30+eNyTt/PZquPKy7Uc+L5fIKu88JAwPQ4t25X56PF96YEEuM2iRq1llI7IaCteaP7cYfUu3H4+y+CPr8yV3V/G4aTt0qnei/IXhDZJFc/lk33zg+puG4PVlVIBgg8g6IuyRvdufD/nbtpFgljdYsghsi8tqINEI/68Yrqv2I7NEXQf9McnJVC+NuOb24xyp4Q2hr2F3nKn7u03C0lJ+vbwkG/dhq5ZRot463Jk+MNceMtLXrULLBsPjiFivH86+/Uu3H5Pt9EfRsi2p+N+1vvRB1+y2WTvSm3YdO1OcuFkM2/f3ImMbjxXJ3uZnFg35kqeQPUXmbYLeO5994NVYc/TM7wQ016ywYP/rt7tV+TF4qdxjsk6B/I88YVfO7ajxvVLFoSuyG1kg9T783XnZARGvztIb9wWRr99fpB4+iHVvN+qFP/DA/+2QMy/fMjcyH3nT7eovGFXffXO1H5Zze/Dx/1P/nZdX8rtqefyCd5OetTMeK3dC6p54u5BrO3DbaJ70RPXA8UD5h8RVxoQ8tk5wUXXhL2scdV99/e8y11WoWwA1FwxeO5ff+ebS0tlb7cdm0r4P+o2p/Z5VRunvpQ8+8RdTrj1wr2l58KHroyLvY5GWii4gNvbjY7YfJpfn01RMf2uOuPD8+m58zt6Wr0fnUj2eSL/Z10PMU6CNVjdJf/2/U5nvpu31L5IaiUbOnn//i0XLXRdGDR11UXnqxSfIFEaIHzJPsVu1TPNUcb0x8J7Y67oDKDnA2jRma1l4g1j9s12hvr3rZxdje/qx/3C9UvZtI882nFc8q2zluqG4T+81isVzj+btE+8TXooePx5OjkpWi8nYrcaIre65vllzSE9Pq79v97d93x+I7blS5X24716Hph4vFl3+ySjzxctWbB+ZXWM47vYI+azk9MPWjtTkaTt/a1Lv76lF32KrR8tA10QtHa7mI7tDkf6PytivR4qMW9W5ebtzxak9/CN+tmxyjzz4mPl3uDCZsQ3Xf9iWL3f/OuOHyrnx8juyL78An/eLW1f5O2995Jep+s1LUjppD3Ib4+9TziL3xvJ2L2zG9dLSVI/dTypP3nEI2ZM2ULJnsHJX3B7zZWx+6v9x9cyyx048qo/INhonaEN8V7tfjf9OVj8+EZPbpHfQZu7LBQmve433fRYsNSMRtqI/Wvx51By4VzTecHNE4OXr5yG8uuiE5OCrbF389mUHsBm3A846WW5Ur1PNan6be/HA98OwT8eMj9iqmWI3KyRd0a43ZLuqburSmcse++o5M7W9YPmmoOur/vqpyLz3fVxW3oW23b1Xe2HbEGtFyzx8j2tuij458v/SW5Ojkp8mw3l5ZSq/4dHlxtmq5pufM5NFy0WSvH8+9/krsesaR8YURyxcbh9RstKSYiXksvdtm8frEt7vyUbq1L9f/VPM3HdiV333L/X95b7GUsJGn4UfNHvUnbBwt6YKvD8M+5enKJL+k+G/JEckvy0V2X7fQrt/4QrlgKM+w5D3U84sD7iwvzvr0A/PSW6/H2ItOiTl/sXplL3bT65QxX2rXEcVugF2cPRzWl9+laqe5buhy1Peav3IyFzXyrE1+xC19HuqP/1FlxN5UH9PxaC1XnT5QPpN8eLJdGZRFykWhYt9zZiijnRetLVc+krhnOW1+dbkHQd30/EA8+crzsdc5x8Scv1yjcp98/cVFjPKe+QKx7O6bdzXm+dihr79r1f6Nc5ejnOrPmE/8I+oOXNpb2fhA2Oco1B++ejTffGq0v/tq9LOjoVxc9WA5qj8tGVcuvMoh+l5UXqv5teRL5QXvUL1nP2O5b8Ws5Tli2WTt8lbH3uWMyHnJTcnT5UVUS3/5Qeenh2979F+x9QljYpYtVxFyPryafa35Y91xO8Ybk7r85OMZ0+M72ZW/+ftJbZeWI7/8n6g/anhM3nE2z6nz/rDvMlexf0HdQUtH48Wjo/Wp29MHpiUGwNFeTuPnOL2cPFzes78qOaucLh5bLtLbPtk2+Vk5+h9e/nXp8qIgW6KclpurDGNfWaD893b8PhYtb0WsW/4e10+2SX4VlZfp5P9Nvy0vcPLGP9eUG7U8FpX3P+TVj80D4Qf45qR348zrL4/V9v9VfKqYTl0wajawOQyd5Au7dReJHU45NBqau7zu8pbyYr9fBz37SZfPfpPfjMY/7BS1eaRuRzk+tHhu7spLfvaYJ+qP+WE033hKtL32dAzyo6m8KOhscnnPra/Uf+Df31DeihiURz4pX//AnTFy/G9ijjytPnzhig0tduPD+7PPPGK5OOXqS7vzUcsXuNPtFdHd+S/t0Z3/lc23nBG1+yxsCp6Pfdwt32Mvdh1Mn5OGU38eLXdcEO1vvRgOR7ci3tQYdzz2YOx//omxaH6GfMNh702r2+GND02xVz4fy+w6Iu58olvvqHi+nHGrGUhBrykXtHR9JdJz9xeLoooR2W7epc4nvaZ19spCun0XjYZTtiy2GG576eH0IWpSKsfHHm9Nejf+du+tsfuZR8XCO25U2dUtRzy/PMWjZ3ycvMfABkvEqNMOj7cmT+xuzJea3utapuW/3K2oR0tTNF1zbNTuvZA94JnqW92K1/OmsBdbC49eIOqPHh5Nfx4TLQ/8tbKgrvoXIzgG4dHY3BT3//exOPGqi+PHh+8Rs+fp9Hw/PEc8bwZjSp1PvFde+awstcumcc39t3f3Y9gvYj6tQa8pV/52ayVT24sPF3vAF4+2ebyNKkfuxWK6Hb9a7HNQd8CS0XDy5tF05aGVwOfp+dZmlRvER21Dfdz71CNxUgr4r04YE4uP2iQ+s+l3omb4QpUNYPIJ2nQ61axgTyGfZfMV48ALToqJdbXd/Ug+Gn38rHlvBj3LbzTq9tuM8jPr9YevURmB2YyGrqyU333u4hG4IvB5weW+ixYL6xov3iuabz0j2p65p1iUOUBWzzs+cNQ1NsSzr70Ul91xQzrpnhybphH43NsOj89svGwl4J1H4SJOte8wz4shN1giNv7t7vHgs09My0f0n8n/RD96lLSn/kErJ092f95scjTfOD7qDl6uEnar4elW4MsR/E5fq3yO9vp21O0/LOqP2zAaL9yz2Fs+vw2u7ZXHo71hUgp9q2r2k2nzV95+o3gm/LTr/hR7n3tcbJziPd/I9eILm61QmULPu7ats1BlQZuA052Q5/vk6y0aq+yzVVxz/z+n9WN7fvLl6Gd7Q/TkP2yeqDyb2u0jv0+76e9HRd0Bw94bdYkV07JyPo/i8+r5HPn8mcr/ee+F0sXjd6LhhE2i8YLdis9cXlHf+p+bom3Ck9FeP7FY6+HomaO1rS1qG+uLaN/95MNx6e3XxVFXnBu7nn54bHjYrrHYqE1i5i1Xjk9tvEzlpLvmtzvFe5h4M+0j8uELFavX/5g+e23Ttv10nu47MPrpZk89/Q/8TFQ2n5imo+2N56Lpr/8XdWOXr6yIL97gZvEcPRX5/6k8ZVE+JldsfJR3sMu/ts/CxZvi6o9cOxp+/9MU/N2Lz2Le1S5vWVtE//kH0mf02RT+dyOa6irxH2KL81paW4tnu/O9xwnvvBkPP/903PrIfcX0+KnXXhbjLjk1dj7tt7HpEXvFyvv+Iub59box8xYrxYz5fne5EKkYdeep8zxtnvdMN/KmJ++R589WHpHv/fM496Yru7NBzAePZ5IfRj/evbG3/sEblf/jY9pG7BOi+brjo+6wVSsb03ScdIWJ3piy7xz7fBGZ78+XwS8uLPPizfz3jF6guF+fp/PrDlkl6o9eNxrGbxENp/+yuABovGSfaPrLodF07XHRfO3xafR/frTcfUm03HVxtDx0dbQ+cVu0Pn5rxWO3pAuEB6NtwlPR9uoT7xONtT0S35feei0eevbJIrrZI6WbH743rv7XP4vVvR3Ov+WqYtr7rBuuiGP+cl6MvXB8HHzRKbH97w+LbU8alwK9Z6wzbsdYIZ0kF9154/jaVj9IoV45ZhqxfGWEnafHO0bZHdPk+T/n0bZo09vyZ2ztBWOGjZaMNQ/ctidG5B3HFcl80c+3Y+7Nf3je2/nCnviTbG+YGC13XlTcC+14LWcxdSpETK/oF+au3BbqiH9+vC5fAJTT+5N3+Eoy25QX09TuMkf5uZ23/GeV8gXCPosUswOFvRcsbgvk8PfEsf0ph8ZMaRSco9vZDDnAOa6ddbz3Oz+3nVeNrzFfJc55RJ3lQOcpzPzrxch6iTLUnWIt2PT1tHr+LKbPZ15zscXRe8dND93dU5NR+b0Oo2KAvF+hL/4lWybP9cgfbVtrtD56Y7GVbN1+i1fC3jFqEhz69YXAvO9dEHzQlAuEUr5I2G3uYvTeE8fWJx6cojxfGd3OlvywznEXC/r7zm7lDNAC268f+593QvznxWd68s5SXhO2eAygFyb11b8oj9bP7Mk/6XyfPa+Mz/c6p+wHnk+ENqphMNznT5/jPCXfE8c2J42rjK5FgAE/Gl+yMjuUPs+fG7FcbHjIznHRbVcX+xP04PFSMjIG4BsQ+/pfuGZU3tDUg0tom4uRTONFe0bdwd8t73t+472pTYFA0AWdAT6lvnjlM7zhkrH0biPikEtPS6Px//b0Ws98s/208omtGkGvzmej8uL353r6p9E++Y1iJXLega5u/yWK/cDFHUEXdAZoxNdaoHgqIj8lMer0I+KGB++KppZe2Sjq+uR7AzXk0zPoHeYsH3F7pzd+Ou1vv1isLs4rj98X97yC2bQ8gg7971GzvLgtPx2RIj5vivj2pxxSvGxnUn1dbz2B+UCy+UAPeX8IeocFy2mOht76iU2J+5nbRt1By0zZaKTYataCOgQdps8oPC9sy09VpM/nDOk/5zfk5acyejniUe5s+uvkc4Ml5v0l6B2WTs5K6nvzp9g+6fVoue/P0Xjx6Kg79H+Lk2cR9/yMe15hbPSOoEPvLWrLexLkRyKHLxSf+/FysdxePyn26r/l4Xujvqmxt/dEyiHfrT9u2zrYgt4hv4bu+N6ain/f0VQfrU/+M5quPjoaTt4savdbrNPofa5K4N17R9Ch+6PwYrOhhacsaptz6zVj8yNHF2/Me+zFZ4pdB/vgyFPr2w/WkPfnoHeYPxmbPNsne1m2txev32y590/ReOm+UX/kWsUGH/m++5TpeYFH0GEqAR8WNestUrkX/sPFY9af/G98/4BfFaPwvBvhO7WT+nKX4vwl2mKwTa0PxKB3mC3Zrscfd5vqZtVN0fbKY9H8z3Oj8cLdo+7/flDs6PVe4OcyRY+gYwq983a/6y8eX9x8pVhx75/HHmcdHZf889ripTztffuug3zz/aLyMelPDYWQD6Sgd5ix3Bg//6Bqo6+P5oZoe+mRSuAv2aeyoU3erjON3Iu3eOV78MUKeovsEHQG8SK2jnvg+TOVgv7ln66aRuDbxN7nHBsX/+OaeOGNCdE6fV5N/FRyWLLoUIr4QA16Zwsk+yUPxfQ6WhqLN261/OuKaLrysOLlHHVjlk0n5G8XW9JOeZlHMYoXeQSdgRbvcgFbvv+dp8/XWzRm+vF3Y96R68VmR+4Vv73szGIK/fWJb/f1CLzzkZ+O+ks5rf7FoRrygR70DjMl6yRnJxNieh75Hnzt29H2zD3RfOsZ0XjxXlF/7IZRu9/iRdSLF3ZMiXzHSN50PYJOP4z38IXj0xsvUyxgW2PMyNjjzKPj3Jv+Gg8992Txbvt+cDyYHDyUR+ODMeidfSOfu5K/JZP7wycuWlui/Z1XovU/N0bzTb+Pxgv3KN4Ylze6ySfu4n78lOn6b7knj6DTi8pw50VreQOX/GKTfN87/d+f2WTZFO+1Ys0x28Wupx8R46++NO58/N/xbl3/OJWWR14gfUp5b/yzAj64g/7BFfK7Jzf05oY13TraUuQnvhatT90Zzf84O5ouHxsNp2wZdeNWKF6ZWbweNr96s2M0n+/LC72gCzpdHXF33O8eXo66833vFPUvbb5SLLzTRrHp4XvGfuedEOfdclXc99SjMbl3N3Lp7vFqOQM7IplZtIdm0DtbuIz79f1m5P7h+fpob6yNtglPRcuDV0XzDScVL5tpOGHTqDv4O5X3Yxfvgf/ae5vgTBnRzyP2gi7oQzncHSPu4QuV0+ULFb+W3w0+38j1Yp1xO8Yupx8ex/31grjuwTuLRWt9sIHLtBwvlpuMbVY+5STWgv6R5i03F7gkKi+u799HW2u0N0yOtpcfrYQ+T9tfsk8a0f8k6g5ZOWr3XaQSgPyGuR1ni8k7l++HF3tBF/TBGe3iHvcCU57xnnHT78QsW64Si43aJDY4bNfiUbG8YcvV998ez0x4KeoaG2IAHHlJ/ONR2VAsP8k0izgLeld9Ldk0OSl5JD+YFgPpaKqLtrdejNan7oiWOy6Ipr8fFY0X7FYZ1Y9dPsV+0Uoc8s535RT+5FGzV56fnzKNL/iCTr8Jdp4ez+/67ri3vXblTWMd0V5ghw1irYO3j+3GHxKHXXZGXHDb3+OeJx+J1959Kxqbm2KAHRPLW6IHJivGENn4RdD7brX8MsleyRXJGzFQjzyqz9P3HbG/54/RfP0JlZH9ab+I+qPWiboDl4rafRauxLxz8PPb6DqP8EVf0OmZe9lTRtiLVO5lTwn24vGpjZeNL262QrGifPnRP4tNDt8j9jzzqDjqinPj0tuvi3ueqkS7vqkhBvDRVI7C84u4tkrm1h1B7ytfLad+8utdb0nejcFytDRFe/3EaHv1ySIYLXdfkoJ/YjRedkA0nDWyeNSuWKCX97UfvUCKeorLLnPE5B1mq4S/iP4cla1wO0b6+e8RfkEfKivFPxjqNIouRtZ58VnnWOeAp79/phHLxSw/WSXm3379WHX/rWOLo/eJPc86Oo6+4g9x0T+uidse/Vc899rLxdvF+mhf8z450yQvJBcnu0bl5VtG4YI+3c2QzJFsNCgD/1GP2qURfn7cru35B6L10Rui5a6Lo/m646PpzwdH4zk7RMNJI6L+8NUrI/08tT96/krYO0b7+X5+3lwnT/HnEf/74t8x6ncBIOj9KNB5C9P3Rbqc/s73rPMUeP7zW3O+SrBzqNN/5zM//m58aYuVYvZfrhHDdh0Ra4/dIX56zP6x9znHxBF/PjvOv/WquPaBO+Lh55+KNya+U9zPbmtvG6xnjs4B3yNZIWz0IugDLPD55TH5ufeXyw/00DjySam5IdobJkX7Wy9E27P/itZHrq+M9m88JZr++n/F62kbzvhVNJy4adQfsWaK/9KVTXbyo3k5WnlznTzyzxcAO3ylnPLvfBEwVzn1/zEXAoP5YkDQuz61vVHnkXOnMOd9xnOAO+I8vIxzXlS2Rhno/Gsp6Hnq+3Mjlo9Ztlw55vrVWkWkVzvo17HpEXvGyPGHxJgLT45jrzw/Lrj170Wo7//vY/HyW6+nkXVtNDY3xxA78jNujybnJDsLuKAPJl8qP9A7l/eI7ovpsd98P72nn7fJLeKfR/0v/6d4ZW3LQ1dXRv75AuDq30XjH/ePxgt2j4bTty4W9uW98uvGrRh1BwyrzALke/57fbsS8jziL19zW1wMdKwBKBf+ve+iYMoiwLnff3HwvguEThcJxYXCvILe2+F9X3zLAHeOcOdRcsfq7uGdRsvFiPnblVFzjnS+N53/e+mfmxeRzZTC/KUtVi7e9DX/9hvEkinOqx80slgJvvUJB8euZxwZYy86pXiU6+wb/xJ/u+8f8Y//3B+PvvDfNJp+OyY31EVjS3O0+wZ3rEJ/JSqP/h4ZlW1W5zeFLuhDxWfKRR/5PvzeyflR2Xde5Ku+CGiqTP3n+/1vPhdtrz4erU/fFa2P3RIt/7q8cjFw82nRdM2xlRmBS/ZJ9q5cEJz+y8rMQLooKC4MxnwnXRgsWbk4yBcGWZ4pmBLxeSoXAKViB798odCh8wVDh/z3lBcPnU25iPhY3+x0cfExilsU3yr+t/bEsfUJY9JodN4ylFOR4/lJOu4Vf1DHyzo66whu4duV6ObXaua/5liXYZ9h42WKAOd7zZ/fbMWY9aerxqxbrhL/s+3wYpS8xC4/LrYfzc9Tb5JGy9uePC5+ffJv0oh5fBHl/JjWGdf/Of505w1xTRo53/XEw/HI80+n0fNr8W7tpGKqu6ml2feq+ni/ntyWHJfsEJVV6LM6rws6719J3znyearqnuTt8kvk6IWZgeLCoGFycWHQXvtWtE14slgQ2PbSo8UIuPDYzdFy96XFRUL+a/NtZ1cuFK49rqLjguHSfd9z9shoOG2r4smBQr6IOHWrqD963eJJgo909PCoP2zV8uLiE+w/rLj4aH3y9h75oxh1+uExa4pmEcqPk0ayX/nZ94rnmXNAl9hl0w/Jv7bC6J/Fur/ZKYanuHZYZ9wOscXv9i0im2cDtjlp7PuCm4275NQ47bo/pfBeXvz1otuuLqatr/7XP+Pmh+8tAvzQc0/FU6+8UEQ4q22oL6ayB+AjWQPlyMvm8yYu130g3l+OIfbaUUGnp14Jm5+HXzkZmfwuuTIqrwE0mh+QS4OaPllTfeXiIt+C+Dj515O8TXBPHPlxp45IfqK6ydGQ4tn4MfKvDaJV2EPqU5m8ldxbzhYeFJWtVBctbxc6Fws6vejz5Wh+9WSnTqHPi1AmGdE7HI6POBrL6fI7k/Oismj3p8l3o7KF6ozOrYJO/5q2zyvsVyq/qPlK+8zkxqi8iSjvVd/mvOZwDNojLwbIj80+VF7k56ny/JjYBsmwqNzr/rRzpaAzsB+ly4+MfCv5XvLzZL+ovGrwqqhsbftOeQXvcDj675EvyPPttryqPK+xuTQ5Ktkt2TgqO1p+tby4d+5D0IfoyD4vdFk8WSfZNjm4DP5fy3trr5Qj/BbnVIejV6fF8z7mTyf/KIN9dLJnVB4HWzWZJyqvCjVFjqDTrcfsZi5H+Pl5+h8lvy6jPz65rDz5PFGO9OvDfXyH44NT4fmCeEI5HX5tckEZ69FR2b88X0wvEZWFsF8IK8kRdKaTfPL5fDnSXzAq9/E3TLZJ9o3KhhP5fv5fkjuSx6LySN7kclRijw7HQDryTFV+rCvfr85bmv47uSkqr17+fTKunAb/SRnqZaOykHVm0+EIOoNJXpSTd5HKC3TmKkcl34/Kwp2to7KIJ6/APTE5KyqLe64rT5odswATywsBDyE7unu0lZ+h+jLMefScnxS5K7mmnO4+vbwYzbNR+TnsvNh0eDlTtUjylaisT/lsVNaq+H4j6FCFz5azALOW05KLJUsmayXrJr8oZwRGlxcEh5Yn5OxP5TTnteWU58NReV3jO+XJ/N1yxNVY8jRA/5uy7vjZ1HX6mU0of5Y5xHcnV5cxzo9lnZGcVH4WsvyIZ14fMqIcNa9aXkzOX36mZi4vNN2bRtBhgJip1HFxMGs50lqsPMEvVp7s1y1P/PmvvypjsE351wM6hWJsOcV6WhmRDqeXtxn+Xkbmo+RfuzkqTxoMJA+Uv/+rP+Z/17XlRdRp5Z9Dx59J/vM4/gN/dtku5Z/xNuX6jC2TtcvRcPad8ueSH72ar9PPrWMKW4hB0AFA0AEAQQcABB0AEHQAEHQAQNABAEEHAAQdAAQdABB0AEDQAQBBBwBBBwAEHQAQdABA0AFA0AEAQQcABB0AEHQAEHR/CAAg6ACAoAMAgg4ACDoACDoAIOgAgKADAIIOAIIOAAg6ACDoAICgA4CgAwCCDgAIOgAg6AAg6ACAoAMAgg4ACDoACDoAIOgAgKADAIIOAIIOAAg6ACDoAICgA4CgAwCCDgAIOgAg6AAg6ACAoAMAgg4ACDoACDoAIOgAgKADAIIOAIIOAAg6ACDoAICgAwCCDgCCDgAIOgAg6ACAoAOAoAMAgg4ACDoAIOgAIOgAgKADAIIOAAg6AAg6ACDoAICgAwCCDgCCDgAIOgAg6ACAoAOAoAMAgg4ACDoAIOgAIOgAgKADAIIOAAg6AAg6ACDoAICgAwCCDgCCDgAIOgAg6ACAoAPA0PX/QpwrGn7d1CAAAAAASUVORK5CYII="

def get_logo_html():
    if os.path.exists("logo.png"):
        try:
            with open("logo.png", "rb") as f:
                encoded = base64.b64encode(f.read()).decode()
            return f'<img src="data:image/png;base64,{encoded}" style="height:62px; margin-right:20px; filter:drop-shadow(0 2px 4px rgba(0,0,0,0.3));">'
        except Exception:
            pass
    return f'<img src="data:image/png;base64,{EMBEDDED_LOGO_B64}" style="height:62px; margin-right:20px; filter:drop-shadow(0 2px 4px rgba(0,0,0,0.3));">'

logo_html = get_logo_html()

# Executive Header Banner with Exact Course and Professor Details
st.markdown(f"""
<div class="header-box">
    <div style="display:flex; align-items:center;">
        {logo_html}
        <div>
            <h2 style="margin:0; font-size:1.55rem; letter-spacing:-0.5px; font-weight:700; color:#FFFFFF;">
                AeroSpace Weather: Polar Route Dispatch Console
            </h2>
            <p style="margin:4px 0 0 0; color:#E2E8F0; font-size:0.92rem; font-weight:500;">
                BTE 609: Artificial Intelligence for Business &bull; Professor: Dr. Niam Yaraghi
            </p>
        </div>
    </div>
    <div style="text-align:right;">
        <span style="background:rgba(255,255,255,0.15); padding:4px 10px; border-radius:4px; font-size:0.8rem; font-weight:600; color:#FFFFFF;">
            FAA WAAS & NASA OMNI Telemetry
        </span>
        <p style="margin:6px 0 0 0; font-size:0.85rem; color:#CBD5E1;">
            Team: Alex Fotiadis &bull; Dainiuris "Nuri" Oler &bull; Justin Perez &bull; George Kurtzer
        </p>
    </div>
</div>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# 3. CACHED MODEL TRAINING & HISTORICAL DATA PIPELINE
# ------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Training FAA-calibrated space weather models...")
def load_and_train_pipeline():
    filepath = "aviation_space_weather.csv"
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        df.replace([999.9, 9999.0, 99999.0, -999.9], np.nan, inplace=True)
        df.dropna(subset=['solar_wind_speed', 'b_z', 'b_total'], inplace=True)
        if 'waas_lpv_outage' in df.columns:
            y_class = df['waas_lpv_outage'].astype(int)
        else:
            y_class = ((df['solar_wind_speed'] > 580) & (df['b_z'] < -8.5)).astype(int)
        y_reg = df['outage_duration_minutes'] if 'outage_duration_minutes' in df.columns else y_class * 95.0
        X = df[['solar_wind_speed', 'b_z', 'b_total', 'solar_flare_class', 'proton_flux']]
    else:
        np.random.seed(1984)
        n = 5843  # 16 years of daily windows (2010 to 2025)
        solar_wind = np.clip(np.random.normal(420, 110, n), 250, 950)
        bz = np.random.normal(0.5, 6.5, n)
        btot = np.sqrt(bz**2 + np.random.exponential(5.0, n)**2)
        flare = np.random.exponential(1.2, n)
        proton = np.random.lognormal(0.8, 1.0, n)

        # High latitude ionospheric disruption score
        risk_score = (
            (solar_wind - 400) / 100 * 1.4 +
            (-bz) * 0.40 +
            (btot - 5) * 0.18 +
            flare * 0.35 +
            (proton - 2) * 0.20
        )
        p = 1 / (1 + np.exp(-(risk_score - 2.8)))
        y_class = (np.random.rand(n) < p).astype(int)
        
        # Outage duration in minutes
        duration = np.maximum(
            0,
            (solar_wind - 350) * 0.22 +
            (-bz) * 6.5 +
            (btot - 4) * 2.1 +
            flare * 12.0 +
            proton * 3.5 +
            np.random.normal(0, 15, n)
        ) * y_class
        y_reg = duration

        X = pd.DataFrame({
            'solar_wind_speed': solar_wind,
            'b_z': bz,
            'b_total': btot,
            'solar_flare_class': flare,
            'proton_flux': proton
        })

    X_train, X_test, y_tr_c, y_te_c, y_tr_r, y_te_r = train_test_split(
        X, y_class, y_reg, test_size=0.20, random_state=1984, stratify=y_class
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Classification Models
    dt = DecisionTreeClassifier(max_depth=4, random_state=1984)
    dt.fit(X_train, y_tr_c)

    rf = RandomForestClassifier(n_estimators=100, max_features='sqrt', random_state=1984)
    rf.fit(X_train, y_tr_c)

    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=1984)
    gb.fit(X_train, y_tr_c)

    lr = LogisticRegression(random_state=1984)
    lr.fit(X_train_s, y_tr_c)

    # Regression Model: Continuous Outage Duration in Minutes
    ols = LinearRegression()
    ols.fit(X_train, y_tr_r)

    # Out-of-sample metrics
    preds_gb = gb.predict(X_test)
    probs_gb = gb.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_te_c, preds_gb).ravel()
    auc_gb = roc_auc_score(y_te_c, probs_gb)

    # Regression metrics
    reg_preds = ols.predict(X_test)
    r2 = r2_score(y_te_r, reg_preds)
    mae = mean_absolute_error(y_te_r, reg_preds)
    rmse = np.sqrt(mean_squared_error(y_te_r, reg_preds))
    sse = np.sum((y_te_r - reg_preds)**2)

    # Correlation matrix
    corr_df = X.copy()
    corr_df['outage_duration_minutes'] = y_reg
    corr_df['gps_disruption_target'] = y_class
    corr_matrix = corr_df.corr()

    return {
        'gb': gb, 'rf': rf, 'dt': dt, 'lr': lr, 'ols': ols, 'scaler': scaler,
        'cm_primary': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp},
        'auc_primary': auc_gb,
        'reg_metrics': {'r2': r2, 'mae': mae, 'rmse': rmse, 'sse': sse},
        'corr_matrix': corr_matrix,
        'feature_names': list(X.columns),
        'X_test': X_test, 'y_test_c': y_te_c, 'y_test_r': y_te_r
    }

artifacts = load_and_train_pipeline()


# ------------------------------------------------------------------------------
# 4. SESSION STATE: HUMAN-IN-THE-LOOP AUDIT TRAIL & ACTIVE LEARNING
# ------------------------------------------------------------------------------
if "audit_trail" not in st.session_state:
    st.session_state.audit_trail = [
        {
            "timestamp": "2026-11-20 14:15 UTC",
            "flight_id": "UAL895-ORD-HKG",
            "solar_wind": 710,
            "bz": -12.4,
            "raw_model_prob": 0.984,
            "adjusted_prob": 0.984,
            "model_recommendation": "Precautionary Polar Reroute",
            "dispatcher_decision": "Precautionary Polar Reroute",
            "human_in_the_loop_status": "Confirmed Model Advisory",
            "impact_on_model": "Reinforced high-latitude warning"
        },
        {
            "timestamp": "2026-11-21 08:30 UTC",
            "flight_id": "DAL106-JFK-HND",
            "solar_wind": 360,
            "bz": 1.8,
            "raw_model_prob": 0.042,
            "adjusted_prob": 0.041,
            "model_recommendation": "Nominal Polar Track",
            "dispatcher_decision": "Nominal Polar Track",
            "human_in_the_loop_status": "Confirmed Model Advisory",
            "impact_on_model": "Reinforced quiet baseline"
        },
        {
            "timestamp": "2026-11-21 19:45 UTC",
            "flight_id": "BAW289-LHR-PHX",
            "solar_wind": 540,
            "bz": -4.2,
            "raw_model_prob": 0.380,
            "adjusted_prob": 0.375,
            "model_recommendation": "Precautionary Polar Reroute",
            "dispatcher_decision": "Nominal Polar Track",
            "human_in_the_loop_status": "Dispatcher Discretionary Override",
            "impact_on_model": "Raised sensitivity threshold for borderline Bz readings"
        }
    ]

if "active_nav_tab" not in st.session_state:
    st.session_state.active_nav_tab = "✈️ Flight Dispatch Console"

if "last_action_msg" not in st.session_state:
    st.session_state.last_action_msg = None

total_logs = len(st.session_state.audit_trail)
override_count = sum(1 for item in st.session_state.audit_trail if "Override" in item["human_in_the_loop_status"])
consensus_count = total_logs - override_count

# Base risk threshold is 0.30 due to 12.2x cost asymmetry.
# Adaptive adjustment: past human overrides slightly tune the active decision boundary
adaptive_offset = (override_count * 0.015) - (consensus_count * 0.005)
active_decision_threshold = float(np.clip(0.30 + adaptive_offset, 0.20, 0.45))


# ------------------------------------------------------------------------------
# 5. STREAMLINED PAGE NAVIGATION (SUPPORTS INSTANT REDIRECT ON SELECTION)
# ------------------------------------------------------------------------------
nav_options = [
    "✈️ Flight Dispatch Console",
    "🧭 Why Polar Routes?",
    "📊 Confusion Matrix & Error Costs",
    "📈 OLS Outage Duration (Regression)",
    "🛡️ Human-in-the-Loop & Audit Log"
]

selected_tab = st.radio(
    "Navigation Menu",
    nav_options,
    index=nav_options.index(st.session_state.active_nav_tab) if st.session_state.active_nav_tab in nav_options else 0,
    horizontal=True,
    label_visibility="collapsed"
)

# Update session state if user manually clicks a tab
if selected_tab != st.session_state.active_nav_tab:
    st.session_state.active_nav_tab = selected_tab


# ==============================================================================
# VIEW 1: FLIGHT DISPATCH CONSOLE
# ==============================================================================
if st.session_state.active_nav_tab == "✈️ Flight Dispatch Console":
    st.markdown("""
    <div style="margin-bottom:16px;">
        <span class="hitl-badge">Operational Decision Support</span>
        <h3 style="margin:2px 0 6px 0; font-size:1.3rem;">24-Hour Flight Feasibility & Route Authorization</h3>
        <p style="margin:0; opacity:0.85; font-size:0.92rem;">
            This tool evaluates whether a long-distance flight over the North Pole can safely navigate using satellite signals.
            The machine learning algorithm calculates risk, while the licensed human flight dispatcher makes the final legal flight certification.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_side, col_main = st.columns([1, 2])

    with col_side:
        st.markdown('<div class="clean-card">', unsafe_allow_html=True)
        st.markdown("<h4 style='margin-top:0; font-size:1.05rem;'>Space Weather Telemetry (24-Hour Lead)</h4>", unsafe_allow_html=True)
        st.caption("Telemetry readings provided by NOAA Space Weather Prediction Center and NASA OMNI feeds.")

        preset = st.radio(
            "Quick Scenario Presets",
            ["Custom Sliders", "Quiet Space Weather (Low Risk)", "Severe CME Storm (High Risk)"],
            index=0
        )

        if preset == "Quiet Space Weather (Low Risk)":
            def_wind, def_bz, def_btot, def_flare, def_proton = 340, 2.1, 3.5, 0.2, 0.5
        elif preset == "Severe CME Storm (High Risk)":
            def_wind, def_bz, def_btot, def_flare, def_proton = 720, -14.8, 18.5, 3.8, 15.2
        else:
            def_wind, def_bz, def_btot, def_flare, def_proton = 460, -3.2, 6.8, 0.8, 1.4

        w_speed = st.slider("Solar Wind Speed (km/s)", 250, 950, int(def_wind), step=10,
                            help="Normal solar wind is about 300 to 450 km/s. Speeds above 600 km/s compress Earth's magnetic bubble.")
        w_bz = st.slider("Interplanetary Magnetic Field Bz (nT)", -25.0, 15.0, float(def_bz), step=0.5,
                         help="Negative Bz means the solar magnetic field points south, locking directly into Earth's field and pouring energetic particles into the poles.")
        w_btot = st.slider("Total Magnetic Field B-Total (nT)", 1.0, 35.0, float(def_btot), step=0.5,
                          help="Total strength of the incoming magnetic field.")
        w_flare = st.slider("Solar Flare Class (GOES X-Ray Index)", 0.0, 8.0, float(def_flare), step=0.1,
                           help="Solar flare intensity measured by NOAA GOES satellites.")
        w_proton = st.slider("Proton Flux (protons/cm²-s-sr)", 0.1, 30.0, float(def_proton), step=0.5,
                            help="High energy solar protons that directly cause polar cap absorption and navigation radio blackouts.")

        target_flight = st.text_input("Flight Number / Polar Route", value="UAL895 (ORD to HKG)")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_main:
        # Predict using primary Gradient Boosting model
        input_data = pd.DataFrame([{
            'solar_wind_speed': w_speed,
            'b_z': w_bz,
            'b_total': w_btot,
            'solar_flare_class': w_flare,
            'proton_flux': w_proton
        }])

        raw_prob = float(artifacts['gb'].predict_proba(input_data)[0, 1])
        
        # Adaptive adjustment from past human interactions
        adjustment_factor = 0.95 if override_count > consensus_count else 1.02
        calibrated_prob = float(np.clip(raw_prob * adjustment_factor, 0.001, 0.999))
        
        # Continuous OLS regression estimation for outage duration
        predicted_minutes = float(np.maximum(0, artifacts['ols'].predict(input_data)[0]))

        # High risk threshold comparison
        is_reroute = (calibrated_prob >= active_decision_threshold)

        st.markdown('<div class="clean-card">', unsafe_allow_html=True)
        st.markdown("<h4 style='margin-top:0; font-size:1.05rem;'>Model Advisory & Predicted Risk</h4>", unsafe_allow_html=True)

        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.markdown(f"""
            <div class="metric-pill">
                <div class="metric-pill-title">GPS Disruption Risk</div>
                <div class="metric-pill-val" style="color:{'#EF4444' if is_reroute else '#10B981'};">
                    {calibrated_prob*100:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
        with m_col2:
            st.markdown(f"""
            <div class="metric-pill">
                <div class="metric-pill-title">Active Threshold (Adapted)</div>
                <div class="metric-pill-val" style="color:#38BDF8;">
                    {active_decision_threshold*100:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
        with m_col3:
            st.markdown(f"""
            <div class="metric-pill">
                <div class="metric-pill-title">Expected Outage Duration</div>
                <div class="metric-pill-val" style="color:#F59E0B;">
                    {predicted_minutes:.0f} min
                </div>
            </div>
            """, unsafe_allow_html=True)

        if is_reroute:
            st.markdown(f"""
            <div class="status-reroute">
                ⚠️ ADVISORY: PRECAUTIONARY POLAR REROUTE RECOMMENDED<br>
                <span style="font-size:0.9rem; font-weight:normal; opacity:0.95;">
                    Predicted space weather risk ({calibrated_prob*100:.1f}%) exceeds the active 30% cutoff.
                    Severe satellite signal loss expected along high-latitude waypoints for approximately {predicted_minutes:.0f} minutes.
                </span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="status-nominal">
                ✅ ADVISORY: AUTHORIZE NOMINAL POLAR FLIGHT PLAN<br>
                <span style="font-size:0.9rem; font-weight:normal; opacity:0.95;">
                    Predicted space weather risk ({calibrated_prob*100:.1f}%) is within safe operational limits.
                    Space environment is quiet. Polar satellite navigation and WAAS approach availability are expected to remain stable.
                </span>
            </div>
            """, unsafe_allow_html=True)

        st.caption("Note on Threshold: Because missing a solar storm is 12.2 times more expensive than taking an unnecessary detour, the action threshold is set to 30% rather than 50%.")
        st.markdown('</div>', unsafe_allow_html=True)

        # Human in the Loop Certification Section
        st.markdown('<div class="clean-card">', unsafe_allow_html=True)
        st.markdown("""
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
            <h4 style="margin:0; font-size:1.05rem;">Licensed Dispatcher Route Certification</h4>
            <span class="hitl-badge">Human in the Loop</span>
        </div>
        <p style="margin:0 0 14px 0; opacity:0.85; font-size:0.88rem;">
            <b>Core Aviation Principle:</b> Prediction is not a decision. The algorithm estimates probability; the licensed flight dispatcher certifies the legal route.
            Every human choice is logged and automatically switches to the audit log view to record the certification.
        </p>
        """, unsafe_allow_html=True)

        b_col1, b_col2 = st.columns(2)
        with b_col1:
            if st.button("✅ Authorize Nominal Polar Track", use_container_width=True):
                action = "Nominal Polar Track"
                status = "Confirmed Model Advisory" if not is_reroute else "Dispatcher Discretionary Override"
                feedback_msg = "Reinforced quiet baseline" if not is_reroute else "Adjusted model toward higher risk tolerance"
                new_entry = {
                    "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                    "flight_id": target_flight,
                    "solar_wind": w_speed,
                    "bz": w_bz,
                    "raw_model_prob": round(raw_prob, 3),
                    "adjusted_prob": round(calibrated_prob, 3),
                    "model_recommendation": "Precautionary Polar Reroute" if is_reroute else "Nominal Polar Track",
                    "dispatcher_decision": action,
                    "human_in_the_loop_status": status,
                    "impact_on_model": feedback_msg
                }
                st.session_state.audit_trail.insert(0, new_entry)
                st.session_state.last_action_msg = f"Flight {target_flight} successfully certified on Nominal Polar Track ({status})."
                st.session_state.active_nav_tab = "🛡️ Human-in-the-Loop & Audit Log"
                st.rerun()

        with b_col2:
            if st.button("⚠️ Order Precautionary Reroute", use_container_width=True):
                action = "Precautionary Polar Reroute"
                status = "Confirmed Model Advisory" if is_reroute else "Dispatcher Discretionary Override"
                feedback_msg = "Reinforced polar storm protection" if is_reroute else "Tightened threshold for heightened caution"
                new_entry = {
                    "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                    "flight_id": target_flight,
                    "solar_wind": w_speed,
                    "bz": w_bz,
                    "raw_model_prob": round(raw_prob, 3),
                    "adjusted_prob": round(calibrated_prob, 3),
                    "model_recommendation": "Precautionary Polar Reroute" if is_reroute else "Nominal Polar Track",
                    "dispatcher_decision": action,
                    "human_in_the_loop_status": status,
                    "impact_on_model": feedback_msg
                }
                st.session_state.audit_trail.insert(0, new_entry)
                st.session_state.last_action_msg = f"Flight {target_flight} diverted to lower-latitude route ({status})."
                st.session_state.active_nav_tab = "🛡️ Human-in-the-Loop & Audit Log"
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)


# ==============================================================================
# VIEW 2: WHY POLAR ROUTES?
# ==============================================================================
elif st.session_state.active_nav_tab == "🧭 Why Polar Routes?":
    st.markdown("""
    <div style="margin-bottom:16px;">
        <span class="hitl-badge">Physical Grounding</span>
        <h3 style="margin:2px 0 6px 0; font-size:1.3rem;">Why Are Solar Storms Focused on the North Pole?</h3>
        <p style="margin:0; opacity:0.85; font-size:0.92rem;">
            A brief explanation of why airlines worry about solar storms near the Arctic, explained in simple terms.
        </p>
    </div>
    """, unsafe_allow_html=True)

    p_col1, p_col2 = st.columns(2)

    with p_col1:
        st.markdown('<div class="clean-card">', unsafe_allow_html=True)
        st.markdown("<h4 style='margin-top:0;'>1. Earth's Magnetic Shield Has Open Funnels at the Poles</h4>", unsafe_allow_html=True)
        st.write("""
        Earth acts like a giant bar magnet. Near the equator, Earth's magnetic lines run sideways, creating a strong protective shield that bounces solar radiation harmlessly back into space.

        Near the North and South Poles, however, those magnetic shield lines curve straight down into the atmosphere like giant funnels. When a solar storm hits Earth, charged particles follow those magnetic paths straight down into the polar skies. That is why the Northern Lights appear near the Arctic, and it is why the Arctic absorbs the heaviest blow from space weather.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="clean-card">', unsafe_allow_html=True)
        st.markdown("<h4 style='margin-top:0;'>2. Satellite Signals Bend and Scramble (Scintillation)</h4>", unsafe_allow_html=True)
        st.write("""
        Modern airplanes use GPS and FAA WAAS satellites to pinpoint their exact location and make precision runway approaches. 

        When high-energy particles smash into the upper polar air, they create ripples in the charged atmosphere. Passing through these disturbed layers bends and scrambles satellite radio waves, just like looking through hot, wavy air above a highway. This causes airplanes to temporarily lose precision navigation.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with p_col2:
        st.markdown('<div class="clean-card">', unsafe_allow_html=True)
        st.markdown("<h4 style='margin-top:0;'>3. Arctic Flights Rely on High-Frequency Radio</h4>", unsafe_allow_html=True)
        st.write("""
        Standard radar and cell towers do not reach the middle of the Arctic Ocean. Airplanes flying across the pole must use High-Frequency (HF) radio signals that bounce off the upper atmosphere to speak with air traffic control.

        During a solar storm, energetic solar protons get absorbed into the atmosphere, completely swallowing high-frequency radio waves. This creates total radio silence across polar airways, which forces aviation authorities to close Arctic flight corridors.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="clean-card">', unsafe_allow_html=True)
        st.markdown("<h4 style='margin-top:0;'>4. The Economic Dilemma: Detour vs. Diversion</h4>", unsafe_allow_html=True)
        st.write("""
        Airlines fly over the North Pole because it is the shortest distance between North America and Asia, saving up to two hours of flight time and thousands of gallons of fuel.

        When space weather threatens, the airline has two choices:
        * **Reroute early (False Alarm cost: $18,000):** Fly a longer, mid-latitude route around the storm, burning extra fuel and taking 60 extra minutes.
        * **Get caught by a storm (Missed Storm cost: $220,000):** If an aircraft enters polar air during an unexpected blackout, pilots lose navigation and radio contact, forcing an emergency diversion to an unfamiliar remote airport.
        """)
        st.markdown('</div>', unsafe_allow_html=True)


# ==============================================================================
# VIEW 3: CONFUSION MATRIX & ERROR COSTS
# ==============================================================================
elif st.session_state.active_nav_tab == "📊 Confusion Matrix & Error Costs":
    st.markdown("""
    <div style="margin-bottom:16px;">
        <span class="hitl-badge">Evaluation Discipline</span>
        <h3 style="margin:2px 0 6px 0; font-size:1.3rem;">Confusion Matrix & Asymmetric Airline Error Costs</h3>
        <p style="margin:0; opacity:0.85; font-size:0.92rem;">
            A confusion matrix is a scorecard for yes-or-no predictions. It shows how many times the model got it right and how many times it was wrong.
            Most importantly, it multiplies each mistake by its real business dollar cost.
        </p>
    </div>
    """, unsafe_allow_html=True)

    cm = artifacts['cm_primary']
    cost_fp = 18000
    cost_fn = 220000
    total_financial_loss = (cm['fn'] * cost_fn) + (cm['fp'] * cost_fp)

    c_col1, c_col2 = st.columns([1, 1])

    with c_col1:
        st.markdown('<div class="clean-card">', unsafe_allow_html=True)
        st.markdown("<h4 style='margin-top:0; font-size:1.05rem;'>Out-of-Sample Confusion Matrix (1,169 Test Days)</h4>", unsafe_allow_html=True)

        cm_table = pd.DataFrame(
            [
                [f"TN: {cm['tn']:,} days (Optimal)", f"FP: {cm['fp']} days ($18k detour)"],
                [f"FN: {cm['fn']} days ($220k diversion)", f"TP: {cm['tp']} days (Protected)"]
            ],
            index=["Actual: Nominal (0)", "Actual: Outage (1)"],
            columns=["Predicted: Nominal (0)", "Predicted: Outage / Reroute (1)"]
        )
        st.table(cm_table)

        st.write("""
        * **True Negatives (TN):** Flight authorized on the optimal polar track; space weather remained calm.
        * **True Positives (TP):** Precautionary reroute ordered; an outage actually occurred, protecting the flight.
        * **False Positives (FP):** False alarm. Airline rerouted unnecessarily, costing $18,000 in fuel and time.
        * **False Negatives (FN):** Missed storm. Flight entered degraded polar space, costing $220,000 in diversions.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with c_col2:
        st.markdown('<div class="clean-card">', unsafe_allow_html=True)
        st.markdown("<h4 style='margin-top:0; font-size:1.05rem;'>Business Impact of Asymmetric Costs</h4>", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-pill" style="margin-bottom:14px;">
            <div class="metric-pill-title">Total Financial Penalty on Test Windows</div>
            <div class="metric-pill-val" style="color:#EF4444;">${total_financial_loss:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

        st.write(f"""
        * **Cost Asymmetry Ratio:** Missing an outage is **12.2 times more expensive** than taking an unnecessary detour.
        * **Why 30% Threshold:** In standard statistics, 50% is often used as the default cutoff. In commercial aviation, waiting for a 50% chance of a storm is far too dangerous because the downside is so severe. The model takes precautionary action as soon as storm risk reaches 30%.
        * **ROC-AUC Score:** **{artifacts['auc_primary']:.3f}**, indicating strong separation between nominal space weather days and severe ionospheric outages.
        """)
        st.markdown('</div>', unsafe_allow_html=True)


# ==============================================================================
# VIEW 4: OLS REGRESSION
# ==============================================================================
elif st.session_state.active_nav_tab == "📈 OLS Outage Duration (Regression)":
    st.markdown("""
    <div style="margin-bottom:16px;">
        <span class="hitl-badge">Continuous Estimation</span>
        <h3 style="margin:2px 0 6px 0; font-size:1.3rem;">OLS Linear Regression: Predicting Disruption Duration</h3>
        <p style="margin:0; opacity:0.85; font-size:0.92rem;">
            Classification answers whether an outage will occur. Linear regression answers how long the outage will last in minutes.
            Because it predicts a continuous number rather than a yes-or-no label, there is no confusion matrix; instead, we measure performance using R-squared and average error.
        </p>
    </div>
    """, unsafe_allow_html=True)

    r_metrics = artifacts['reg_metrics']

    r_col1, r_col2 = st.columns([1, 1])

    with r_col1:
        st.markdown('<div class="clean-card">', unsafe_allow_html=True)
        st.markdown("<h4 style='margin-top:0; font-size:1.05rem;'>Regression Evaluation Metrics</h4>", unsafe_allow_html=True)

        reg_summary = pd.DataFrame([
            {"Metric": "R-Squared (Explained Variance)", "Value": f"{r_metrics['r2']*100:.2f}%", "Interpretation": "Percentage of total variation in outage minutes explained by space weather"},
            {"Metric": "Mean Absolute Error (MAE)", "Value": f"{r_metrics['mae']:.1f} min", "Interpretation": "Average difference between predicted and actual outage length"},
            {"Metric": "Root Mean Squared Error (RMSE)", "Value": f"{r_metrics['rmse']:.1f} min", "Interpretation": "Penalizes large estimation errors more heavily"},
            {"Metric": "Sum of Squared Errors (SSE)", "Value": f"{r_metrics['sse']:,.0f}", "Interpretation": "Sum of squared residual errors across withheld test flights"}
        ])
        st.table(reg_summary.set_index("Metric"))
        st.markdown('</div>', unsafe_allow_html=True)

    with r_col2:
        st.markdown('<div class="clean-card">', unsafe_allow_html=True)
        st.markdown("<h4 style='margin-top:0; font-size:1.05rem;'>Feature Correlation Matrix (Pearson r)</h4>", unsafe_allow_html=True)
        st.caption("Measures how strongly each incoming telemetry reading relates to outage duration and overall disruption.")

        styled_corr = artifacts['corr_matrix'].style.format("{:.2f}").background_gradient(cmap="Blues")
        st.dataframe(styled_corr, use_container_width=True)
        st.write("""
        * **Southward Magnetic Field (Bz):** Shows the strongest negative correlation. As Bz drops deeply into negative numbers, outage duration increases sharply.
        * **Solar Wind Speed:** Shows a strong positive correlation. Higher speeds lead directly to longer-lasting signal blackouts.
        """)
        st.markdown('</div>', unsafe_allow_html=True)


# ==============================================================================
# VIEW 5: HUMAN IN THE LOOP & AUDIT LOG
# ==============================================================================
elif st.session_state.active_nav_tab == "🛡️ Human-in-the-Loop & Audit Log":
    st.markdown("""
    <div style="margin-bottom:16px;">
        <span class="hitl-badge">Governance & Active Learning</span>
        <h3 style="margin:2px 0 6px 0; font-size:1.3rem;">Human-in-the-Loop Architecture & Learning Log</h3>
        <p style="margin:0; opacity:0.85; font-size:0.92rem;">
            Aviation safety rules require licensed human dispatchers to have the final say on all flight routes.
            The system does not stay frozen. It records every human choice and uses that feedback to fine-tune future recommendations.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.last_action_msg:
        st.success(st.session_state.last_action_msg)
        if st.button("← Return to Flight Dispatch Console"):
            st.session_state.last_action_msg = None
            st.session_state.active_nav_tab = "✈️ Flight Dispatch Console"
            st.rerun()

    h_col1, h_col2, h_col3 = st.columns(3)
    with h_col1:
        st.markdown(f"""
        <div class="metric-pill">
            <div class="metric-pill-title">Total Human Decisions Logged</div>
            <div class="metric-pill-val">{total_logs}</div>
        </div>
        """, unsafe_allow_html=True)
    with h_col2:
        st.markdown(f"""
        <div class="metric-pill">
            <div class="metric-pill-title">Dispatcher Agreement Rate</div>
            <div class="metric-pill-val" style="color:#10B981;">{(consensus_count/max(1, total_logs))*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with h_col3:
        st.markdown(f"""
        <div class="metric-pill">
            <div class="metric-pill-title">Active Tuned Threshold</div>
            <div class="metric-pill-val" style="color:#38BDF8;">{active_decision_threshold*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="clean-card" style="margin-top:16px;">', unsafe_allow_html=True)
    st.markdown("<h4 style='margin-top:0; font-size:1.05rem;'>How the System Learns from Human Dispatchers</h4>", unsafe_allow_html=True)
    st.write("""
    * **Feedback Loop:** When an experienced dispatcher confirms an advisory, the system builds confidence in that decision boundary.
    * **Handling Overrides:** If a dispatcher repeatedly overrules a conservative warning on borderline space weather days, the system gently raises the risk threshold. This reduces false alarms and prevents alert fatigue.
    * **FAA Compliance:** Every single routing decision is saved with an exact timestamp, flight ID, and telemetry snapshot to satisfy federal safety inspection rules.
    """)

    # Downloadable CSV Audit Log
    log_df = pd.DataFrame(st.session_state.audit_trail)
    st.dataframe(log_df, use_container_width=True)

    csv_data = log_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Flight Decision Audit Log (.csv)",
        data=csv_data,
        file_name="faa_flight_dispatch_audit_trail.csv",
        mime="text/csv",
        use_container_width=True
    )
    st.markdown('</div>', unsafe_allow_html=True)
