# CSV AI Chatbot 🤖

An AI-powered chatbot that allows users to interact with CSV datasets using natural language. Instead of manually exploring rows, columns, and statistics, users can upload a CSV file and ask questions about their data to receive relevant, AI-generated responses.

## 🚀 Features

- **CSV File Upload:** Upload CSV datasets directly through the Streamlit interface.
- **Natural Language Queries:** Ask questions about your dataset using simple English.
- **AI-Powered Responses:** Uses Google's Gemini API to understand questions and generate relevant answers.
- **Data Processing:** Uses Pandas to read, process, and analyze uploaded datasets.
- **Dynamic Dataset Support:** Works with different CSV datasets instead of being restricted to a predefined dataset.
- **Interactive Interface:** Provides a simple and user-friendly interface built with Streamlit.

## 🛠️ Tech Stack

- **Programming Language:** Python
- **Data Processing:** Pandas
- **AI Integration:** Google Gemini API
- **Frontend:** Streamlit

## ⚙️ How It Works

1. Upload a CSV file through the application.
2. The application reads and processes the dataset using Pandas.
3. Enter a question about the uploaded data in natural language.
4. The Gemini API helps interpret the query and generate a relevant response.
5. View the generated answer through the Streamlit interface.

## 📂 Project Structure

```text
CSV-AI-Chatbot/
│
├── app.py
├── requirements.txt
├── .env
└── README.md
```

## 💻 Installation and Setup

**1. Clone the repository**

```bash
git clone <your-repository-url>
cd CSV-AI-Chatbot
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Configure your Gemini API key**

Create a `.env` file in the project directory and add:

```env
GEMINI_API_KEY=your_api_key_here
```

**4. Run the application**

```bash
streamlit run app.py
```

The application will open in your browser.

## 🔮 Future Improvements

- Support for additional file formats such as Excel.
- Data visualization through automatically generated charts.
- Advanced statistical analysis.
- Improved handling of complex queries.
- Support for larger datasets.

## 👩‍💻 Author

Developed as a Python and Generative AI project to explore how Large Language Models can simplify data analysis through natural language interaction.
