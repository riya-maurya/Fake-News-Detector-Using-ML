import streamlit as st
import pickle
import numpy as np

from wordcloud import WordCloud
import matplotlib.pyplot as plt


# EXAMPLES TO TRY
example_real = """WASHINGTON (Reuters) - A Georgian-American businessman who met then-Miss Universe pageant owner Donald Trump in 2013, has been questioned by congressional investigators about whether he helped organize a meeting between Russians and Trumpâ€™s eldest son during the 2016 election campaign, four sources familiar with the matter said. The meeting at Trump Tower in New York involving Donald Trump Jr. and other campaign advisers is a focus of probes by Congress and Special Counsel Robert Mueller on whether campaign officials colluded with Russia when it sought to interfere in the U.S. election, the sources said. Russia denies allegations by U.S. intelligence agencies that it meddled in the election and President Donald Trump denies any collusion. The Senate and House of Representatives intelligence committees recently questioned behind closed doors Irakly Kaveladze, a U.S. citizen born in the former Soviet republic of Georgia, the sources said. He is a U.S.-based representative of Azerbaijani oligarch Aras Agalarovâ€™s real estate firm, the Crocus Group. The panels knew Kaveladze was at the June 9, 2016 meeting but became more interested in him after learning he also attended a private dinner in Las Vegas in 2013 with Trump and Agalarov as they celebrated an agreement to hold that yearâ€™s Miss Universe pageant in Moscow, the sources said.  Committee members now want to know more about the extent of Kaveladzeâ€™s contacts with the Trump family and whether he had a bigger role than previously believed in setting up the Trump Tower meeting when Trump was a Republican candidate for president. The White House declined to comment. Muellerâ€™s office also declined to comment. Scott Balber, a New York lawyer who represents Kaveladze, confirmed that his client attended both the dinner in Las Vegas and the Trump Tower meeting but said he did not set up the second meeting. Trumpâ€™s son-in-law Jared Kushner, other Trump campaign aides, and Russian lawyer Natalia Veselnitskaya were also at that meeting. Lawyer Balber also said the committees were only seeking Kaveladzeâ€™s input as a witness and were not targeting him for investigation. â€œNo-one has ever told me that they have any interest in him other than as a witness,â€ Balber said. Lawyers for Trump Jr. and Kushner did not respond to requests for comment about their contacts with Kaveladze.Â A lawyer for President Trump declined to comment. One photograph from the 2013 dinner, when Trump still owned the Miss Universe pageant, shows Agalarov and his pop singer son Emin along with Trump, two Trump aides and several other people at the dining table. Another shows Kaveladze standing behind Trump and Emin Agalarov as they speak. The pictures were found by a University of California at Irvine student and blogger Scott Stedman, who posted them on Nov. 22. Aras Agalarov is a billionaire property developer in Russia who was awarded the Order of Honor by Russian President Vladimir Putin. Several U.S. officials who spoke on condition of anonymity said Muellerâ€™s team and the committees are looking for any evidence of a link between the Trump Tower meeting and the release six weeks later of emails stolen from Democratic Party organizations. They are also trying to determine whether there was any discussion at the New York meeting of lifting U.S. economic sanctions on Russia, a top priority for Putin, the officials said. Rob Goldstone, a British publicist, told Trump Jr. ahead of the New York meeting that Russian lawyer Veselnitskaya would be bringing damaging information about donations to a charity linked to Trumpâ€™s Democratic rival Hillary Clinton, according to emails later released by Trump Jr. Trump Jr. initially said the meeting was about Russian adoptions but later said it also included Veselnitskayaâ€™s promises of information on the donations to the Clinton charity. He said he ultimately never received the information, although it was later posted on the Internet. In a statement issued after meeting with the Senate Judiciary Committee on Sept. 7, Trump, Jr. said Goldstone and Veselnitskaya were in a conference room with him as well as Kaveladze and a translator. Balber said Kaveladze attended expecting to serve as a translator, although he did not do so in the end because Veselnitskaya brought her own. """
example_fake = """Alabama is a notoriously deep red state. It s a place where Democrats always think that we have zero chances of winning   especially in statewide federal elections. However, that is just what happened on Tuesday night in the Special Election to replace Senator Jeff Sessions. Doug Jones, the Democratic Senate candidate who is known in the state for prosecuting the Ku Klux Klan members who bombed a church during the Civil Rights Movement and killed four little African American girls, will be the next Senator from Alabama. CNN has just called the race, as there seems no more GOP-leaning counties out there.To contrast, Roy Moore had been twice removed from the Alabama Supreme Court as Chief Justice for violating the law, and was also credibly accused of being a sexual predator toward teen girls. Despite all of that, though, the race was a nail biter, because Moore has a long history and a deep base in Alabama. Of course, decent people   including Republicans   were horrified at the idea of a man like Roy Moore going to the Senate. Despite the allegations of sexual predation, Moore also had said many incendiary things, such as putting forth the idea that Muslims shouldn t be allowed in Congress, that homosexuality should be illegal, and that America was great when slavery was legal. And that s just for starters, too.Thank you Alabama, for letting sanity prevail in this race. Oh, and a message to Democrats   this is proof we can compete everywhere. Get a fifty state strategy going so we can blow the GOP outta the water in 2018.Featured image via Justin Sullivan/Getty Images"""

# Set up the app
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

# Load model and vectorizer with error handling
@st.cache_resource
def load_components():
    try:
        with open('RandomForest.pkl', 'rb') as f:  # UPDATED filename here
            model = pickle.load(f)
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

model, vectorizer = load_components()

# Prediction function with debugging
def predict(text):
    try:
        features = vectorizer.transform([text])
        prediction = model.predict(features)[0]
        proba = model.predict_proba(features)[0]
        
        labels = {0: 'FAKE', 1: 'REAL'}  # Make sure this matches your training label encoding
        predicted_label = labels.get(prediction, 'UNKNOWN')
        confidence = proba[prediction] * 100
        
        return predicted_label, confidence, proba
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        return None, None, None

# Visualization functions
def create_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig

def create_probability_chart(proba):
    labels = ['FAKE', 'REAL']  # Make sure these match your model
    fig, ax = plt.subplots()
    bars = ax.bar(labels, proba, color=['red', 'green'])
    ax.set_ylim(0, 1)
    ax.set_title('Prediction Confidence')
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height,
                f'{height:.2f}',
                ha='center', va='bottom')
    return fig

# Main app
def main():
    st.title(" Fake News Detector 🕵️")
    st.markdown("Analyze news articles for authenticity using machine learning")
    
    tab1, tab2 = st.tabs(["Detector", "About"])
    
    with tab1:
        st.header("Check News Authenticity")
        
        st.markdown("**Try these examples:**")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Example REAL"):
                st.session_state['input_text'] = example_real
        with col2:
            if st.button("Example FAKE"):
                st.session_state['input_text'] = example_fake

        news_text = st.text_area("Paste news content:", height=200,
                                 placeholder="Enter article text or headline here...",
                                 value=st.session_state.get('input_text', ''))

        if st.button("Analyze", type="primary"):
            if not news_text.strip():
                st.warning("Please enter some text to analyze")
            else:
                with st.spinner("Processing..."):
                    label, confidence, proba = predict(news_text)
                    
                    if label == 'FAKE':
                        st.error(f"🚨 FAKE NEWS (confidence: {confidence:.1f}%)")
                    elif label == 'REAL':
                        st.success(f"✅ REAL NEWS (confidence: {confidence:.1f}%)")
                    else:
                        st.warning("Could not determine authenticity")
                    
                    if proba is not None:
                        st.pyplot(create_probability_chart(proba))
                    
                    st.subheader("Text Analysis")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.pyplot(create_wordcloud(news_text))
                    with col2:
                        st.write(f"**Word count:** {len(news_text.split())}")
                        st.write(f"**Character count:** {len(news_text)}")
    
    with tab2:
        st.header("About This Tool")
        st.markdown("""
         **Fake News Detector** is an AI-powered web application designed to detect misinformation in news content using machine learning techniques.

    ### How It Works
    - The tool is powered by a trained **Random Forest classifier**
    - It uses **TF-IDF vectorization** to convert text into numerical features
    - The model predicts whether a news article is more likely to be **REAL** or **FAKE**
    - It also displays **confidence scores** and **visualizations** to help interpret the result

    ### Key Features
    - **Headline & Content Analysis**: Paste any article text or headline to evaluate its authenticity
    - **Confidence Score Visualization**: View how confident the model is in its prediction using a probability bar chart
    - **Word Cloud**: Instantly visualize the most frequent terms from your input
    - **Text Summary Stats**: Get word and character count of the article
    - **Interactive & Easy to Use**: Built with Streamlit for a smooth and responsive experience

    ---
    ### Contact Me
    Have questions, ideas, or just want to connect? I'd love to hear from you!

    - **Name:** Riya Maurya  
    - **Email:** [rheamaurya8826@gmail.com](mailto:rheamaurya8826@gmail.com)  
    - **LinkedIn:** [linkedin.com/in/riya-maurya](https://linkedin.com/in/riya-maurya)  
    - **GitHub:** [github.com/riya-maurya](https://github.com/riya-maurya)

    ---
    *This app is built for academic and educational purposes. Accuracy depends on the quality and diversity of training data.*
    """)

if __name__ == "__main__":
    main()