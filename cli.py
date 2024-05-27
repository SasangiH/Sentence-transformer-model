#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('run', '"./run_script.ipynb"')


# In[ ]:


# Main function to handle CLI
def main():
    # Define your sentences
    sentences = [
        "In recent years, technology has advanced at an unprecedented rate, revolutionizing various aspects of our lives. From the advent of smartphones and high-speed internet to the development of artificial intelligence and machine learning algorithms, innovations in technology have transformed the way we communicate, work, and access information. With the rapid pace of innovation, it's essential for individuals and organizations to adapt to these changes to stay competitive in today's digital landscape.",
        "Climate change poses significant challenges to our planet, affecting ecosystems, weather patterns, and human livelihoods. Rising global temperatures, melting ice caps, and extreme weather events are just some of the consequences of climate change that we are witnessing. Urgent action is needed to mitigate the impacts of climate change through measures such as reducing greenhouse gas emissions, transitioning to renewable energy sources, and implementing sustainable practices in agriculture and industry.",
        "Healthcare innovation has led to remarkable advancements in medical treatments, diagnostics, and patient care. Breakthroughs in areas such as precision medicine, gene editing, and telemedicine are revolutionizing the way we prevent, diagnose, and treat diseases. These innovations have the potential to improve patient outcomes, reduce healthcare costs, and enhance the overall quality of life for individuals around the world.",
        "The rise of e-commerce has transformed the retail industry, offering consumers convenience, choice, and accessibility like never before. Online shopping platforms have become increasingly popular, allowing people to browse and purchase products from the comfort of their homes. With the proliferation of mobile devices and digital payment systems, e-commerce is expected to continue growing, reshaping traditional retail models and consumer behavior.",
        "Education plays a crucial role in shaping individuals' lives and driving societal progress. It provides knowledge, skills, and opportunities for personal and professional growth, empowering individuals to reach their full potential. Access to quality education is essential for fostering innovation, economic development, and social cohesion. By investing in education, societies can build a brighter future for generations to come.",
        "Healthy foods play a pivotal role in maintaining optimal health and well-being. Incorporating a balanced diet rich in fruits, vegetables, whole grains, lean proteins, and healthy fats provides essential nutrients, vitamins, and minerals that support overall health. Fresh fruits and vegetables are abundant sources of antioxidants, fiber, and phytonutrients, which help boost the immune system, reduce inflammation, and protect against chronic diseases such as heart disease, diabetes, and cancer. Whole grains provide sustained energy, promote digestive health, and help regulate blood sugar levels. Lean proteins such as fish, poultry, beans, and legumes are essential for muscle repair, metabolism, and satiety. Healthy fats found in nuts, seeds, avocados, and olive oil are crucial for brain function, hormone production, and heart health. By prioritizing nutrient-dense foods and adopting a balanced eating pattern, individuals can nourish their bodies and enhance their overall quality of life.",
        "Mental health awareness has gained traction as societies recognize the importance of addressing mental health issues and destigmatizing mental illness. Increased awareness, advocacy, and access to mental health services have helped reduce barriers to treatment and support for individuals struggling with mental health challenges. Promoting mental well-being and providing support systems are essential steps toward creating communities that prioritize mental health and emotional resilience.",
        "The younger generation contends with elevated levels of stress stemming from academic pressure, social expectations, economic uncertainties, and digital saturation. Academic pursuits fuel intense competition and anxiety, while the pervasive influence of social media amplifies feelings of inadequacy and comparison. Economic challenges, coupled with personal and societal pressures, compound stress for young adults. To address this, society must acknowledge and support the younger generation, providing resources and coping strategies to navigate stressors and cultivate resilience.",
        "Treatment for mental illness encompasses a holistic approach involving therapy, medication, lifestyle adjustments, and support networks. Psychotherapy helps address negative thought patterns, while medications alleviate symptoms and stabilize mood. Lifestyle modifications such as exercise and stress management contribute to overall well-being. Additionally, strong support networks and community resources play a vital role in providing social and emotional support. A personalized treatment plan combining these approaches is typically most effective in managing mental illness.",
        "Cultural diversity enriches societies by celebrating differences in language, religion, customs, and traditions. Embracing cultural diversity promotes tolerance, understanding, and mutual respect among individuals from different backgrounds. Inclusive policies and practices that value diversity contribute to social cohesion, economic prosperity, and innovation. By fostering an inclusive society, we can build a more equitable and harmonious world for all."
    ]

    # Preprocess documents
    vocabulary, word_to_index, preprocessed_sentence = preprocess_sentences(sentences)

    # Generate embeddings
    sentence_vectors = vector_embeddings(preprocessed_sentence)

    # Store embeddings in VectorStore
    vector_store = store_embeddings(sentence_vectors)

    # Start CLI loop
    while True:
        query_sentence = input("Enter your query (type 'exit' to quit): ")

        if query_sentence.lower() == 'exit':
            print("Exiting...")
            break
        else:
            query_vector = model.encode(query_sentence)
            query_tokens = preprocess_query(query_sentence)

            for token in query_tokens:
              if token in word_to_index:
                query_vector[word_to_index[token]] += 1

            #similar_sentences = searching_for_similarity(vector_store, query_sentence, word_to_index)
            similar_sentences = vector_store.find_similar_vectors(query_vector, num_results=3)
            print("Query Sentence:", query_sentence)
            print("Similar Sentences:")
            for sentence, similarity in similar_sentences:
                print(f"{sentence}: Similarity = {similarity.item():.4f}")

if __name__ == "__main__":
    main()



# In[ ]:




