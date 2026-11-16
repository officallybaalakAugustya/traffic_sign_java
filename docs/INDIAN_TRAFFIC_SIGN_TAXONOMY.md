# Indian Traffic Sign Taxonomy: IRC:67 Code of Practice and Class Mapping

**Project:** Indian Traffic Sign Recognition and Road Safety Monitor  
**Standard Authority:** Indian Roads Congress (IRC:67) & Ministry of Road Transport and Highways (MoRTH)  
**Taxonomy Version:** 1.0.0  
**Machine-Readable Mapping:** `src/main/resources/taxonomies/indian_traffic_signs_irc67.json`  
**Date:** October 19, 2026  

---

## 1. Legal and Regulatory Framework

In India, traffic signs are governed under **IRC:67: Code of Practice for Road Signs**, issued by the Indian Roads Congress and enacted under **Section 116 of the Motor Vehicles Act, 1988**.

IRC:67 standardizes the geometric shape, color palettes, retroreflective sheeting grades, symbols, and bilingual/trilingual lettering across all National Highways, State Highways, and Urban Roads in India.

### The Three Functional Sign Categories

| Category | Legal Nature | Dominant Geometry | Dominant Color Palette | Safety Function |
| :--- | :--- | :--- | :--- | :--- |
| **1. Mandatory / Regulatory Signs** | **Legally Enforceable.** Violation constitutes an offence under the Motor Vehicles Act. | Circular (Exceptions: Octagon for *Stop*, Inverted Triangle for *Give Way*). | White background, red border, black symbol, red diagonal slash. *Compulsory positive instructions* use circular blue discs with white arrows/symbols. | Enforces movement prohibitions, directional obligations, vehicle class restrictions, and speed caps. |
| **2. Cautionary / Warning Signs** | **Advisory / Warning.** Alerts drivers to road hazards ahead. | Equilateral Triangle with apex pointing upwards. | White background, red border, black pictogram. | Prompts defensive driving, speed reduction, and lane discipline before road hazards. |
| **3. Informatory / Guide Signs** | **Navigational / Informational.** Aids navigation and public convenience. | Rectangular or Square boards. | Blue background (amenities/expressways), green background (state/national highways), or white with black text. | Guides route selection and identifies roadside amenities (hospitals, fuel stations, parking). |

---

## 2. Indian Road Peculiarities Represented in Taxonomy

Unlike European standards (GTSRB / Vienna Convention), IRC:67 introduces signs specific to Indian road ecology:
1. **Cattle Crossing (`CATTLE`):** Accounts for roaming livestock on rural and peri-urban highways.
2. **Speed Breaker / Hump Road (`HUMP_OR_ROUGH_ROAD`):** Critical for warning of harsh rumble strips and unauthorized speed breakers that cause severe vehicular rollover risks.
3. **Horn Prohibited (`HORN_PROHIBITED`) vs Compulsory Sound Horn (`COMPULSARY_SOUND_HORN`):** Diametrically opposite rules; silence zones around hospitals vs mandatory acoustic warning on blind Himalayan/Western Ghat curves.
4. **Gap in Median (`GAP_IN_MEDIAN`):** Alerts of authorized and unauthorized median openings on multi-lane divided highways where sudden U-turns occur.
5. **Slow Transport Prohibitions:** `BULLOCK_PROHIBITED`, `HANDCART_PROHIBITED`, and `CYCLE_PROHIBITED` separating high-speed motor traffic from vulnerable non-motorized transport.

---

## 3. Systematic Class Inventory & Safety Mapping

The primary Indian dataset (`kannanwisen/Indian-Traffic-Sign-Classification`) encompasses 85 classes directly mapped to the IRC:67 standard:

### A. Mandatory Signs (Regulatory & Prohibitory)

| Internal ID | Dataset Label | Canonical Name | Geometry | Speed Cap | Safety Decision Support Alert |
| :---: | :--- | :--- | :--- | :---: | :--- |
| **0** | `ALL_MOTOR_VEHICLE_PROHIBITED` | All Motor Vehicles Prohibited | Circular Red Border | — | **DANGER:** Non-motorized zone or pedestrian precinct. Do not enter. |
| **1** | `AXLE_LOAD_LIMIT` | Axle Load Limit | Circular Red Border | — | **CRITICAL:** Heavy vehicle weight limit. Bridge collapse risk. |
| **3** | `BULLOCK_PROHIBITED` | Bullock Carts Prohibited | Circular Red Border | — | Slow animal transport prohibited. |
| **5** | `COMPULSARY_AHEAD` | Compulsory Ahead Only | Circular Blue Disc | — | **WARNING:** No turns permitted. Maintain forward lane. |
| **6** | `COMPULSARY_AHEAD_OR_TURN_RIGHT` | Ahead or Turn Right | Circular Blue Disc | — | Lane maneuver restricted to straight or right. |
| **7** | `COMPULSARY_CYCLE_TRACK` | Compulsory Cycle Track | Circular Blue Disc | — | Motor vehicles keep off cycle track. |
| **8** | `COMPULSARY_KEEP_LEFT` | Compulsory Keep Left | Circular Blue Disc | — | **CRITICAL:** Divider/island ahead. Keep left to avoid median impact. |
| **9** | `COMPULSARY_KEEP_RIGHT` | Compulsory Keep Right | Circular Blue Disc | — | Construction diversion. Pass obstacle on the right. |
| **10** | `COMPULSARY_MINIMUM_SPEED` | Compulsory Minimum Speed | Circular Blue Disc | — | Express lane minimum speed. Increase speed to avoid rear-end collision. |
| **11** | `COMPULSARY_SOUND_HORN` | Compulsory Sound Horn | Circular Blue Disc | — | **SAFETY ACTION:** Blind curve/ghat. Sound acoustic horn to alert oncoming traffic. |
| **12** | `COMPULSARY_TURN_LEFT` | Compulsory Turn Left | Circular Blue Disc | — | Mandatory left exit / one-way entry. |
| **13** | `COMPULSARY_TURN_RIGHT_AHEAD` | Compulsory Turn Right Ahead | Circular Blue Disc | — | Advance turn requirement. |
| **15** | `CYCLE_PROHIBITED` | Cycles Prohibited | Circular Red Border | — | Cyclists barred from expressway/flyover. |
| **20** | `GIVE_WAY` | Give Way / Yield | Inverted Triangle | — | **YIELD:** Yield right-of-way to main road traffic. Prepare to stop. |
| **22** | `HANDCART_PROHIBITED` | Handcarts Prohibited | Circular Red Border | — | Manual cart restriction. |
| **23** | `HEIGHT_LIMIT` | Height Limit | Circular Red Border | — | **CRITICAL:** Low clearance underpass/overhead wire ahead. |
| **24** | `HORN_PROHIBITED` | Horn Prohibited | Circular Red Border | — | **SILENCE ZONE:** Hospital/court area. Acoustic horns strictly banned. |
| **27** | `LEFT_TURN_PROHIBITED` | Left Turn Prohibited | Circular Red Slash | — | No left turn into opposing traffic stream. |
| **31** | `NO_ENTRY` | No Entry | Circular Solid Red | — | **FATAL RISK:** Wrong-way entry strictly forbidden. Stop immediately. |
| **32** | `NO_PARKING` | No Parking | Circular Red Slash | — | Vehicle stopping for loading/unloading only. |
| **33** | `NO_STOPPING_OR_STANDING` | No Stopping or Standing | Circular Red Cross | — | High-speed clearance lane. Zero stopping permitted. |
| **34** | `OVERTAKING_PROHIBITED` | Overtaking Prohibited | Circular Red Slash | — | **DANGER:** Two-way undivided section. Do not pull out to overtake. |
| **37** | `PEDESTRIAN_PROHIBITED` | Pedestrians Prohibited | Circular Red Slash | — | Access-controlled expressway. |
| **38** | `PRIORITY_FOR_ONCOMING_VEHICLES` | Priority for Oncoming Traffic | Circular Red Border | — | Narrow bridge/passage. Yield to oncoming vehicle. |
| **40** | `RESTRICTION_ENDS` | Restriction Ends | Circular Black Slash | — | Prior regulatory limits lifted. |
| **48** | `SPEED_LIMIT_5` | Speed Limit (5 km/h) | Circular Red Border | 5 | Crawl speed: gate / security barrier. |
| **49** | `SPEED_LIMIT_15` | Speed Limit (15 km/h) | Circular Red Border | 15 | Slow speed: steep ramp / parking structure. |
| **50** | `SPEED_LIMIT_30` | Speed Limit (30 km/h) | Circular Red Border | 30 | School / market speed cap. |
| **51** | `SPEED_LIMIT_40` | Speed Limit (40 km/h) | Circular Red Border | 40 | Standard urban arterial speed limit. |
| **52** | `SPEED_LIMIT_50` | Speed Limit (50 km/h) | Circular Red Border | 50 | Urban sub-arterial limit. |
| **53** | `SPEED_LIMIT_60` | Speed Limit (60 km/h) | Circular Red Border | 60 | City bypass / ring road limit. |
| **54** | `SPEED_LIMIT_70` | Speed Limit (70 km/h) | Circular Red Border | 70 | State highway limit. |
| **55** | `SPEED_LIMIT_80` | Speed Limit (80 km/h) | Circular Red Border | 80 | National highway passenger limit. |
| **59** | `STOP` | Stop | Octagonal Red | — | **MANDATORY STOP:** Full halt behind stop line. Check all directions. |
| **60** | `STRAIGHT_PROHIBITED` | Straight Prohibited | Circular Red Slash | — | No through way. |
| **62** | `TRUCK_PROHIBITED` | Trucks Prohibited | Circular Red Slash | — | Commercial freight diversion. |
| **63** | `TURN_RIGHT` | Turn Right | Circular Blue Disc | — | Mandatory right direction. |
| **64** | `U_TURN_PROHIBITED` | U-Turn Prohibited | Circular Red Slash | — | **DANGER:** Median U-turn banned due to high collision hazard. |

---

### B. Cautionary / Warning Signs (Equilateral Triangles)

| Internal ID | Dataset Label | Canonical Name | Road Hazard Description | Safety Decision Support Alert |
| :---: | :--- | :--- | :--- | :--- |
| **2** | `BARRIER_AHEAD` | Barrier Ahead | Toll barrier, police checkpoint, or railway gate | Decelerate; anticipate full stop ahead. |
| **4** | `CATTLE` | Cattle Crossing | Free-roaming livestock grazing on verges | **HAZARD:** Reduce speed; watch shoulders for cattle darting. |
| **14** | `CROSS_ROAD` | Cross Road | Uncontrolled 4-way intersection | Scan left and right; yield to cross traffic. |
| **16** | `DANGEROUS_DIP` | Dangerous Dip | Sharp road depression / water causeway | Slow down to prevent vehicle undercarriage scraping or swerving. |
| **18** | `FALLING_ROCKS` | Falling Rocks | Landslide and loose boulder zone | Watch cliff side; avoid stopping under rock overhangs. |
| **19** | `GAP_IN_MEDIAN` | Gap in Median | Cut in central highway median | **HIGH COLLISION RISK:** Expect vehicles executing abrupt U-turns. |
| **21** | `GUARDED_LEVEL_CROSSING` | Guarded Railway Crossing | Train crossing with gates | Slow down; prepare to stop if red signals flash. |
| **25** | `HUMP_OR_ROUGH_ROAD` | Speed Breaker / Rough Road | Speed breaker or deteriorated pavement | **CRITICAL HAZARD:** Decelerate immediately to prevent loss of control. |
| **26** | `LEFT_REVERSE_BEND` | Left Reverse Bend | Double curve (left then right) | Decelerate; stay strictly within lane markings. |
| **28** | `LOOSE_GRAVEL` | Loose Gravel | Unbound stone chips on surface | Reduce speed; prevent skidding and stone projection onto windshields. |
| **29** | `MEN_AT_WORK` | Men at Work | Highway maintenance personnel in lane | Slow down; watch for roadside flaggers and heavy machinery. |
| **30** | `NARROW_ROAD_AHEAD` | Narrow Road Ahead | Carriageway constricts | Merge smoothly; do not attempt overtaking. |
| **36** | `PEDESTRIAN_CROSSING` | Pedestrian Crossing | Zebra crossing ahead | **VULNERABLE USERS:** Yield to pedestrians on roadway. |
| **39** | `QUAY_SIDE_OR_RIVER_BANK` | River Bank / Quay Side | Unprotected drop into water body | Heighten vigilance; avoid night driving close to unbarricaded edges. |
| **41** | `RIGHT_HAIR_PIN_BEND` | Right Hairpin Bend | 180° hairpin turn on mountain ghats | Downshift gear; sound horn; negotiate at low speed. |
| **42** | `RIGHT_HAND_CURVE` | Right Hand Curve | Sharp right curve | Decelerate; centrifugal drift warning. |
| **43** | `ROUNDABOUT` | Roundabout Ahead | Traffic rotary ahead | Yield right-of-way to vehicles already in roundabout. |
| **44** | `SCHOOL_AHEAD` | School Ahead | School zone with children crossing | **CHILDREN:** Reduce speed strictly to $\le 25\text{ km/h}$. |
| **45** | `SIDE_ROAD_LEFT` | Side Road Left | Minor road joins from left | Watch for joining vehicles. |
| **46** | `SIDE_ROAD_RIGHT` | Side Road Right | Minor road joins from right | Watch for joining vehicles. |
| **47** | `SLIPPERY_ROAD` | Slippery Road | Wet, oily, or polished road surface | Increase following distance; avoid sudden braking. |
| **56** | `STAGGERED_INTERSECTION` | Staggered Intersection | Two offset side roads | Expect turning cross traffic in sequence. |
| **57** | `STEEP_ASCENT` | Steep Ascent | Uphill road gradient | Select lower gear to maintain torque. |
| **58** | `STEEP_DESCENT` | Steep Descent | Downhill road gradient | Use engine braking; avoid prolonged brake friction. |
| **61** | `T_INTERSECTION` | T-Intersection | Road terminates perpendicularly | Prepare to stop and turn left or right. |
| **65** | `Y_INTERSECTION` | Y-Intersection | Road bifurcates into two branches | Select branch lane in advance. |

---

### C. Informatory Signs (Rectangular / Amenity)

| Internal ID | Dataset Label | Canonical Name | Functional Purpose |
| :---: | :--- | :--- | :--- |
| **17** | `DIRECTION` | Direction / Navigational Arrow | Route guidance at highway forks. |
| **35** | `PASS_EITHER_SIDE` | Pass Either Side | Permitted traffic flow to either side of bollard. |

---

## 4. Unknown and Unmapped Class Handling Protocol

In real-world deployment across Indian states, models will encounter:
- Newly posted regional municipal signs.
- Heavily defaced, rusted, or partially occluded boards.
- Non-sign objects mimicking sign geometries (e.g., circular red advertisement logos, triangular warning reflectors on stationary trucks).

To prevent overconfident false classifications, our pipeline defines:
1. **Internal ID `-1` / Label `"UNKNOWN_SIGN"`:** Returned whenever the Softmax prediction confidence falls below an empirical threshold ($\tau < 0.65$) or when the predicted class is not present in the verified taxonomy.
2. **Entropy Rejection:** Shannon entropy $H(p) = -\sum p_i \log p_i$ is evaluated over output probabilities. Uniform uncertainty ($H(p) \approx \log 85$) flags an ambiguous or out-of-distribution signpost.
3. **Graceful Fallback:** If an unmapped label string is encountered during dataset ingestion, it is recorded in `ClassFrequencyReport.unmappedLabels()` rather than halting execution or silently assigning class `0`.
