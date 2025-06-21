import os
import requests
import base64
from typing import List, Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class UdemyCourseRecommender:
    def __init__(self, test_mode: bool = True):
        self.test_mode = test_mode
        self.base_url = 'https://www.udemy.com?srsltid=AfmBOooax65Bdwjf9Itr0M_Kcs58r58Q_TadZKWub_OPXGaLG_DuzX6T/api-2.0/'
        
        # Mock data for testing
        self.mock_data = {
            'python': [
                {'title': 'Python Bootcamp 2024', 'url': '/python-bootcamp', 'rating': 4.8, 'price': '$13.99'},
                {'title': 'Python for Data Science', 'url': '/python-data', 'rating': 4.6, 'price': '$12.99'}
            ],
            'javascript': [
                {'title': 'Modern JavaScript', 'url': '/modern-js', 'rating': 4.7, 'price': '$11.99'},
                {'title': 'Full Stack JavaScript', 'url': '/fullstack-js', 'rating': 4.5, 'price': '$14.99'}
            ],
            'sql': [
                {'title': 'SQL Masterclass', 'url': '/sql-masterclass', 'rating': 4.7, 'price': '$10.99'}
            ],
            'aws': [
                {'title': 'AWS Certified Solutions Architect', 'url': '/aws-architect', 'rating': 4.8, 'price': '$15.99'}
            ],
            'docker': [
                {'title': 'Docker for Beginners', 'url': '/docker-beginners', 'rating': 4.6, 'price': '$9.99'}
            ],
            # ... thêm các kỹ năng khác nếu muốn
        }

        if not test_mode:
            self.client_id = os.getenv('UDEMY_CLIENT_ID')
            self.client_secret = os.getenv('UDEMY_CLIENT_SECRET')
            if not all([self.client_id, self.client_secret]):
                print("Warning: Missing API credentials. Switching to test mode.")
                self.test_mode = True
            else:
                self.headers = self._get_headers()

    def _get_headers(self) -> Dict:
        credentials = f"{self.client_id}:{self.client_secret}"
        encoded = base64.b64encode(credentials.encode()).decode()
        return {
            'Authorization': f'Bearer {encoded}',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }

    def search_courses(self, skill: str, limit: int = 3) -> List[Dict]:
        if self.test_mode:
            return self.mock_data.get(skill.lower(), [])[:limit]
        
        try:
            endpoint = f"{self.base_url}courses/"
            params = {
                'search': skill,
                'page_size': limit,
                'ordering': 'highest-rated',
                'ratings_gte': 4.0,
                'fields[course]': 'title,url,price,rating'
            }
            response = requests.get(endpoint, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json().get('results', [])
        except Exception as e:
            print(f"Error fetching courses: {e}")
            return self.mock_data.get(skill.lower(), [])[:limit]

    def get_recommendations(self, skills: List[str], limit: int = 5) -> List[Dict]:
        all_courses = []
        for skill in skills:
            courses = self.search_courses(skill)
            formatted = [{
                'title': course['title'],
                'url': course['url'],
                'skill': skill,
                'rating': course.get('rating', 'N/A'),
                'price': course.get('price', 'N/A')
            } for course in courses]
            all_courses.extend(formatted)
        return all_courses[:limit]

if __name__ == '__main__':
    recommender = UdemyCourseRecommender(test_mode=True)
    skills = ['python', 'javascript']
    recommendations = recommender.get_recommendations(skills)
    print("\nCourse Recommendations:")
    for course in recommendations:
        print(f"\nTitle: {course['title']}")
        print(f"Skill: {course['skill']}")
        print(f"Rating: {course['rating']}")
        print(f"Price: {course['price']}")