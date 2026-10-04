"""The benchmark set shipped with fclass, so `fclass bench` works anywhere.

60 realistic document previews, 5 per category, with generic or misleading
filenames so a classifier must read the content. Several are deliberately
confusable (insurance vs. bills, tax receipts vs. statements, manuals vs.
technical references). Each item is (filename, label, text).
"""

# The fixed taxonomy the benchmark is scored against (independent of your config).
CATEGORIES = {
    "Finance/Statements": "Bank statements, credit card bills, utility bills, invoices, payment receipts",
    "Finance/Taxes": "Tax returns, W-2, 1099, Form 16, tax receipts and deduction proofs",
    "Finance/Investments": "Brokerage and portfolio statements, stock trades, retirement and pension accounts",
    "Study/Courses": "Syllabi, assignments, course materials, exam papers, certificates of completion",
    "Study/Notes": "Personal notes, lecture notes, highlights, study guides, journals",
    "Study/References": "Research papers, technical documentation, textbooks, reference material",
    "Recreation/Hobbies": "Hobby projects, recipes, patterns, collections, DIY guides",
    "Recreation/Entertainment": "Ebooks, articles, saved web content, games, movies and music lists",
    "Recreation/Travel": "Itineraries, flight and hotel bookings, tickets, visas, trip plans",
    "Keep/Important": "IDs, passports, birth certificates, wills, contracts, leases, insurance policies, legal documents. These go here even when they mention money.",
    "Keep/Archives": "Old or completed projects and historical records worth preserving",
    "Keep/Manuals": "Product manuals, warranties, setup and instruction guides",
}

DATASET = [
    # ── Finance/Statements ─────────────────────────────────────────────────
    ("scan_0042.pdf", "Finance/Statements",
     "HDFC BANK  Statement of Account  Period: 01/03/2025 to 31/03/2025\nAccount No: XXXXXX4821  Savings\n"
     "Date  Narration  Withdrawal  Deposit  Closing Balance\n02/03 UPI-SWIGGY 412.00  48,211.50\n"
     "05/03 NEFT-SALARY ACME CORP  1,20,000.00  1,68,211.50\n09/03 ATM WDL 5,000.00 1,63,211.50"),
    ("document(3).pdf", "Finance/Statements",
     "Your electricity bill. Pacific Gas and Electric. Service for: 412 Elm St. Billing period Feb 12 - Mar 13.\n"
     "Amount due: $142.18  Due date: April 2. Total usage 612 kWh. Compare your usage to last year."),
    ("Untitled.pdf", "Finance/Statements",
     "INVOICE #INV-20931\nBill to: Lalit S.\nFrom: Figma, Inc.\nDescription: Professional plan (annual) Qty 1  $144.00\n"
     "Tax $0.00  Total $144.00  Paid via Visa ending 1188 on 14 Jan 2025. Thank you for your business."),
    ("export.csv", "Finance/Statements",
     "Transaction Date,Post Date,Description,Category,Type,Amount\n01/04/2025,01/05/2025,AMAZON MKTPL,Shopping,Sale,-23.99\n"
     "01/06/2025,01/07/2025,TRADER JOE'S,Groceries,Sale,-64.12\n01/09/2025,01/09/2025,Payment Thank You,,Payment,500.00"),
    ("IMG_receipt.txt", "Finance/Statements",
     "Payment receipt. Airtel Postpaid. Mobile 98XXXXXX21. Bill amount Rs 799.00 paid successfully on 03-Feb-2025 "
     "via UPI. Transaction ID 50349211. Next bill date 01-Mar-2025."),

    # ── Finance/Taxes ──────────────────────────────────────────────────────
    ("scan_0043.pdf", "Finance/Taxes",
     "Form W-2 Wage and Tax Statement 2024. Employer identification number 12-3456789. Wages, tips, other compensation "
     "98,500.00. Federal income tax withheld 14,210.00. Social security wages 98,500.00. Copy B To Be Filed With Employee's FEDERAL Tax Return."),
    ("final_v2.pdf", "Finance/Taxes",
     "FORM NO. 16 [See rule 31(1)(a)] PART A Certificate under section 203 of the Income-tax Act, 1961 for tax deducted "
     "at source on salary. Assessment Year 2025-26. TAN of the Deductor: BLRA12345F. Summary of amount paid and tax deducted."),
    ("donation.pdf", "Finance/Taxes",
     "Thank you for your gift. Red Cross acknowledges receipt of your contribution of $250.00 on Dec 18, 2024. "
     "No goods or services were provided in exchange. Please retain this letter for your tax deduction records under IRC 170."),
    ("notes.txt", "Finance/Taxes",
     "TODO for filing: get 1099-INT from Ally, 1099-B from Schwab, property tax receipt, HSA 5498. "
     "Standard vs itemized? Estimated quarterly payment Q4 was $3,000. Accountant appointment March 3."),
    ("download.pdf", "Finance/Taxes",
     "Income Tax Department. Acknowledgement Number 3021456789. ITR-1 SAHAJ filed for Assessment Year 2024-25. "
     "Gross total income 14,20,000. Total tax payable 1,12,500. Refund due 8,400. Verified electronically via Aadhaar OTP."),

    # ── Finance/Investments ────────────────────────────────────────────────
    ("statement.pdf", "Finance/Investments",
     "Charles Schwab  Brokerage Account Statement  Q1 2025. Account Value $84,212.55. Positions: VTI 120 shares, "
     "AAPL 15 shares, SCHD 200 shares. Realized gain/loss YTD +1,204.10. Dividends received 312.44."),
    ("scan_0044.pdf", "Finance/Investments",
     "Consolidated Account Statement (CAS) NSDL/CDSL. Mutual Fund holdings: Parag Parikh Flexi Cap Fund - Direct Growth "
     "units 1,203.441 NAV 78.12. SIP registered Rs 10,000 monthly. Portfolio XIRR 16.2%."),
    ("doc.pdf", "Finance/Investments",
     "Fidelity NetBenefits 401(k) quarterly statement. Your employer match this quarter: $2,150. Vested balance $61,880. "
     "Investment elections: Fidelity Freedom 2055 Fund 100%. Rate of return year to date 6.1%."),
    ("trade_confirm.txt", "Finance/Investments",
     "Trade Confirmation. Bought 10 NVDA @ 118.42 on NASDAQ. Settlement date T+1. Commission 0.00. Order type LIMIT. "
     "Account ending 7731. Robinhood Securities LLC."),
    ("crypto.csv", "Finance/Investments",
     "Timestamp,Transaction Type,Asset,Quantity Transacted,Spot Price at Transaction,Subtotal\n"
     "2024-11-02,Buy,BTC,0.012,69420.11,833.04\n2024-12-15,Buy,ETH,0.5,3870.22,1935.11\n2025-01-20,Sell,BTC,0.004,102331.0,409.32"),

    # ── Study/Courses ──────────────────────────────────────────────────────
    ("week3.pdf", "Study/Courses",
     "CS 229 Machine Learning — Problem Set 2. Due: Wednesday Oct 23 at 11:59pm via Gradescope. 1. [15 points] Logistic "
     "regression: derive the Hessian of the log-likelihood and show it is negative semidefinite."),
    ("syllabus.docx", "Study/Courses",
     "ECON 101 Principles of Microeconomics, Fall 2024. Instructor: Dr. Patel. Office hours Tue 2-4pm. Grading: Midterm 30%, "
     "Final 40%, Problem sets 20%, Participation 10%. Week 1: Supply and demand."),
    ("certificate.pdf", "Study/Courses",
     "Coursera. This is to certify that Lalit Shewani successfully completed Deep Learning Specialization, a 5-course "
     "series offered by DeepLearning.AI. Verify at coursera.org/verify/ABC123. Andrew Ng, Instructor."),
    ("scan_0045.pdf", "Study/Courses",
     "Mid-term Examination. Course: Product Management Fundamentals (PM-201). Time allowed: 2 hours. Answer any FOUR questions. "
     "Q1. Explain the difference between output and outcome metrics with an example."),
    ("Untitled(1).docx", "Study/Courses",
     "Assignment 4 submission — Group 7. Topic: Market sizing for a B2B SaaS product in India. TAM/SAM/SOM analysis. "
     "Submitted to Prof. Rao, Strategy course, IIM module 3."),

    # ── Study/Notes ────────────────────────────────────────────────────────
    ("New Text Document.txt", "Study/Notes",
     "lecture 7 notes - attention\n- query/key/value, scaled by sqrt(d_k)\n- multi-head = parallel subspaces\n"
     "- why positional enc? no recurrence!\n? ask TA about rotary vs sinusoidal\nremember: complexity O(n^2)"),
    ("notes.md", "Study/Notes",
     "# Book highlights — Thinking, Fast and Slow\n> System 1 operates automatically and quickly\n- anchoring: first number "
     "biases estimates\n- my takeaway: write down estimates BEFORE seeing data"),
    ("scratch.txt", "Study/Notes",
     "studying for GMAT quant:\nweak areas -> combinatorics, rate problems\nformula: work = rate x time\n"
     "practice set 3 got 11/15. redo q 4, 9, 12 tomorrow"),
    ("journal.md", "Study/Notes",
     "## 2025-02-11\nLearned today how vector DBs use HNSW graphs. My summary: navigable small world layers, greedy search "
     "from top layer down. Need to compare with IVF-PQ next week."),
    ("doc1.txt", "Study/Notes",
     "Spanish practice notes\nser vs estar: ser = permanent, estar = temporary/location\nconjugation drill: yo soy, tú eres\n"
     "vocab: la cuenta = the bill, el viaje = the trip"),

    # ── Study/References ───────────────────────────────────────────────────
    ("2401.04088.pdf", "Study/References",
     "Abstract. We introduce Mixtral 8x7B, a Sparse Mixture of Experts (SMoE) language model. Mixtral has the same "
     "architecture as Mistral 7B, with the difference that each layer is composed of 8 feedforward blocks (i.e. experts)."),
    ("scan_0046.pdf", "Study/References",
     "PostgreSQL 16 Documentation. Chapter 11. Indexes. 11.2 Index Types. PostgreSQL provides several index types: "
     "B-tree, Hash, GiST, SP-GiST, GIN, BRIN. Each index type uses a different algorithm."),
    ("book.pdf", "Study/References",
     "Designing Data-Intensive Applications. Martin Kleppmann. Chapter 5: Replication. Leaders and Followers. "
     "Each node that stores a copy of the database is called a replica."),
    ("whitepaper.pdf", "Study/References",
     "McKinsey Global Institute report: The economic potential of generative AI. Executive summary. Generative AI could "
     "add the equivalent of $2.6 trillion to $4.4 trillion annually across the 63 use cases we analyzed."),
    ("api.md", "Study/References",
     "## Rate limits\nThe API enforces requests-per-minute and tokens-per-minute limits per organization. When exceeded, "
     "the API returns HTTP 429 with a retry-after header. Use exponential backoff."),

    # ── Recreation/Hobbies ─────────────────────────────────────────────────
    ("scan_0047.pdf", "Recreation/Hobbies",
     "Sourdough loaf. Ingredients: 500g bread flour, 350g water, 100g active starter, 10g salt. Autolyse 1 hr, "
     "stretch and folds every 30 min x4, bulk ferment until 50% rise, cold retard overnight."),
    ("plan.txt", "Recreation/Hobbies",
     "Raised garden bed build: 4x8 ft cedar, 2x10 boards, corner posts 4x4. Soil mix 60% topsoil 30% compost 10% perlite. "
     "Plant tomatoes, basil, marigolds as companions in April."),
    ("pattern.pdf", "Recreation/Hobbies",
     "Beginner amigurumi bunny crochet pattern. Materials: 4mm hook, worsted yarn, safety eyes. Rnd 1: 6 sc in magic ring (6). "
     "Rnd 2: inc in each st around (12)."),
    ("doc2.pdf", "Recreation/Hobbies",
     "Arduino weather station project. Parts: ESP32, BME280 sensor, 0.96 OLED. Wiring: SDA to GPIO21, SCL to GPIO22. "
     "Upload sketch, readings every 10 minutes to my home dashboard."),
    ("list.xlsx", "Recreation/Hobbies",
     "Sheet1\nStamp, Country, Year, Condition, Acquired\nPenny Red, UK, 1864, Fine, eBay\nGandhi 10R, India, 1948, Mint, "
     "Dad's album\nInverted Jenny replica, USA, 1918, Good, fair"),

    # ── Recreation/Entertainment ───────────────────────────────────────────
    ("download(2).pdf", "Recreation/Entertainment",
     "Chapter One. The rain had not stopped for three days when Mara found the letter wedged beneath the door, "
     "its envelope soft as cloth and addressed in her dead brother's hand."),
    ("watchlist.txt", "Recreation/Entertainment",
     "To watch: Shōgun (FX), Severance S2, Past Lives, Dune Part Two, The Bear S3.\nWatched: Oppenheimer 9/10, "
     "Poor Things 7/10. Ask Riya for anime recs."),
    ("article.html", "Recreation/Entertainment",
     "The 25 best video games of the decade so far — ranked. From Elden Ring's open world to the cozy charm of "
     "Animal Crossing, our critics pick the games that defined the 2020s."),
    ("scan_0048.pdf", "Recreation/Entertainment",
     "Coldplay Music of the Spheres World Tour. Admit one. DY Patil Stadium, Navi Mumbai. 19 Jan 2025, 7:00 PM. "
     "Gate 4, Stand B. This e-ticket is non-transferable."),
    ("playlist.csv", "Recreation/Entertainment",
     "Track,Artist,Album,Duration\nKhaab,Akhil,Single,3:41\nBlinding Lights,The Weeknd,After Hours,3:20\n"
     "Kesariya,Arijit Singh,Brahmastra,4:28"),

    # ── Recreation/Travel ──────────────────────────────────────────────────
    ("eticket.pdf", "Recreation/Travel",
     "Electronic Ticket Itinerary/Receipt. Passenger: SHEWANI/LALIT MR. Booking ref Q7XK2P. AI 191 Mumbai (BOM) to "
     "Newark (EWR) 14 Jun 2025 01:30, Economy, Baggage 2PC. Seat 34A."),
    ("confirmation.pdf", "Recreation/Travel",
     "Booking.com. Your booking is confirmed. Hotel Arts Barcelona. Check-in Fri 12 Sep 2025 from 15:00. Check-out Mon "
     "15 Sep 2025 until 12:00. 2 adults. Free cancellation until 5 Sep."),
    ("trip.md", "Recreation/Travel",
     "Japan trip plan\nDay 1 Tokyo: Shibuya, Meiji shrine\nDay 3: Shinkansen to Kyoto, JR pass activate\n"
     "Day 5: Nara deer park\nBudget: ~¥30k/day. Book teamLab tickets in advance."),
    ("scan_0049.pdf", "Recreation/Travel",
     "Schengen Visa application appointment confirmation. VFS Global, Bengaluru. Applicant: Lalit Shewani. "
     "Purpose of travel: Tourism. Documents: passport, flight reservation, hotel booking, bank statement 6 months."),
    ("doc3.txt", "Recreation/Travel",
     "Packing list - Ladakh bike trip: thermals, riding gloves, rain cover, diamox, power bank, permits for Nubra and "
     "Pangong (inner line permit), spare tube, cash (ATMs scarce)."),

    # ── Keep/Important ─────────────────────────────────────────────────────
    ("scan_0050.pdf", "Keep/Important",
     "LIFE INSURANCE POLICY DOCUMENT. Policy No. 99812734. Policyholder: Lalit Shewani. Sum Assured Rs 1,00,00,000. "
     "Annual Premium Rs 14,200. Nominee: Mrs. S. Shewani. Policy term 30 years. Terms and conditions attached."),
    ("lease.pdf", "Keep/Important",
     "RESIDENTIAL LEASE AGREEMENT. This Agreement is made between John Miller (Landlord) and Lalit Shewani (Tenant) for "
     "the premises at 221 Park Ave, Apt 5B. Term: 12 months commencing July 1. Monthly rent $2,400. Security deposit $4,800."),
    ("ID.pdf", "Keep/Important",
     "REPUBLIC OF INDIA  PASSPORT  Type P  Country Code IND  Passport No. Z1234567  Surname SHEWANI  Given Name LALIT  "
     "Date of Birth 01/01/1990  Place of Issue MUMBAI  Date of Expiry 12/05/2033"),
    ("offer.pdf", "Keep/Important",
     "EMPLOYMENT AGREEMENT. This agreement is entered into between Acme Technologies Pvt Ltd and the Employee. "
     "Compensation: CTC Rs 45,00,000 per annum. Notice period 60 days. Non-compete and confidentiality clauses apply."),
    ("scan_0051.pdf", "Keep/Important",
     "Health insurance e-card and policy schedule. Star Health Family Floater. Insured members: 3. Sum insured 10 lakh. "
     "Premium paid Rs 22,410 incl GST. Policy period 01-Apr-2025 to 31-Mar-2026. Cashless hospital network."),

    # ── Keep/Archives ──────────────────────────────────────────────────────
    ("old_project.md", "Keep/Archives",
     "# Project Falcon — final retrospective (completed Dec 2021)\nWhat went well: shipped on time. What didn't: scope creep "
     "in Q3. This project is closed; repo archived. Keeping for reference."),
    ("scan_0052.pdf", "Keep/Archives",
     "Class X Board Examination 2006 Statement of Marks. Central Board of Secondary Education. English 88, Mathematics 95, "
     "Science 91. Result: PASS. Roll No 1234567."),
    ("letter.txt", "Keep/Archives",
     "Dear Lalit, Congratulations on completing five years with us (2015-2020). As you move on, we wish you the best. "
     "This relieving letter confirms your last working day as 30 Sep 2020. HR, Infosys."),
    ("history.docx", "Keep/Archives",
     "Family tree notes compiled by grandpa in 1998. Shewani family originally from Hyderabad, Sindh. Moved to Ulhasnagar in "
     "1948. Names and dates of great-grandparents below."),
    ("v1_final.pptx", "Keep/Archives",
     "Slide 1: Q4 2019 Board Review (FINAL, superseded). Slide 2: Revenue grew 22%. Slide 3: Roadmap 2020. "
     "[Archived deck from previous company]"),

    # ── Keep/Manuals ───────────────────────────────────────────────────────
    ("scan_0053.pdf", "Keep/Manuals",
     "Samsung Front Load Washing Machine WW90T User Manual. Safety information. Installation: remove the transit bolts "
     "before use. Error code 4C: water supply problem. Warranty: 2 years comprehensive, 10 years on motor."),
    ("guide.pdf", "Keep/Manuals",
     "Quick Start Guide. TP-Link Archer AX55 Wi-Fi 6 router. 1. Power off your modem. 2. Connect the router to the modem "
     "via Ethernet. 3. Scan the QR code to set up with the Tether app. Default SSID on the label."),
    ("warranty.pdf", "Keep/Manuals",
     "Limited Warranty Card. Product: Dyson V15 Detect. Serial number: SV22-XXXX. Date of purchase: 02/11/2024. "
     "Register your machine within 30 days to activate your 2 year guarantee on parts and labour."),
    ("doc4.pdf", "Keep/Manuals",
     "IKEA MALM chest of 6 drawers assembly instructions. You will need: flat screwdriver, hammer. Step 1: attach "
     "rails 104321. Important: must be anchored to the wall to prevent tip-over."),
    ("manual.txt", "Keep/Manuals",
     "Instant Pot Duo 7-in-1 owner's manual. Pressure cooking time chart: rice 3 min, chicken breast 10 min. "
     "Do not fill above the PC MAX line. Clean the steam release valve regularly."),
]
