import numpy as np
import librosa
import os
import json
import pickle
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
import time
import warnings
warnings.filterwarnings('ignore')

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

try:
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import classification_report
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import RandomForestClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class SafeJSONEncoder:
    """Безпечний JSON енкодер для уникнення circular reference"""
    
    @staticmethod
    def encode_value(obj):
        """Безпечна конвертація значень для JSON"""
        if obj is None:
            return None
        elif isinstance(obj, (bool, int, float, str)):
            return obj
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (list, tuple)):
            return [SafeJSONEncoder.encode_value(item) for item in obj]
        elif isinstance(obj, dict):
            return {str(key): SafeJSONEncoder.encode_value(value) for key, value in obj.items()}
        elif isinstance(obj, defaultdict):
            return {str(key): SafeJSONEncoder.encode_value(value) for key, value in dict(obj).items()}
        elif hasattr(obj, '__dict__'):
            # Для об'єктів з атрибутами - перетворюємо в dict
            return {str(key): SafeJSONEncoder.encode_value(value) 
                   for key, value in obj.__dict__.items() 
                   if not key.startswith('_') and not callable(value)}
        else:
            return str(obj)


class ComprehensiveDatasetValidator:
    """Комплексний валідатор для аудіо та FFT датасетів"""
    
    def __init__(self, sr=44100):
        self.sr = sr
        self.validation_results = {
            'audio_validation': {},
            'fft_validation': {},
            'cross_validation': {},
            'recommendations': []
        }
        
    def validate_complete_pipeline(self, base_path, generate_reports=True):
        """Валідація всього пайплайну обробки"""
        print("КОМПЛЕКСНА ВАЛІДАЦІЯ ДАТАСЕТУ")
        print("=" * 60)
        
        results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'base_path': str(base_path)
        }
        
        try:
            # 1. Валідація аудіосегментів
            segments_path = os.path.join(base_path, "processed-segments")
            if os.path.exists(segments_path):
                print("\nВАЛІДАЦІЯ АУДІОСЕГМЕНТІВ")
                print("-" * 40)
                audio_results = self.validate_audio_dataset(segments_path)
                results['audio_validation'] = audio_results
            else:
                print("Аудіосегменти не знайдено")
                results['audio_validation'] = {'status': 'not_found'}
            
            # 2. Валідація FFT датасету
            fft_path = os.path.join(base_path, "fft-dataset")
            if os.path.exists(fft_path):
                print("\nВАЛІДАЦІЯ FFT ДАТАСЕТУ")
                print("-" * 40)
                fft_results = self.validate_fft_dataset(fft_path)
                results['fft_validation'] = fft_results
            else:
                print("FFT датасет не знайдено")
                results['fft_validation'] = {'status': 'not_found'}
            
            # 3. Крос-валідація між датасетами
            if (results['audio_validation'].get('status') == 'success' and 
                results['fft_validation'].get('status') == 'success'):
                print("\nКРОС-ВАЛІДАЦІЯ ДАТАСЕТІВ")
                print("-" * 40)
                cross_results = self.cross_validate_datasets(segments_path, fft_path)
                results['cross_validation'] = cross_results
            
            # 4. Генерація рекомендацій
            recommendations = self.generate_comprehensive_recommendations(results)
            results['recommendations'] = recommendations
            
            # 5. Збереження звітів
            if generate_reports:
                self.generate_comprehensive_report(results, base_path)
            
            self.print_final_summary(results)
            
        except Exception as e:
            print(f"Помилка під час валідації: {e}")
            results['error'] = str(e)
        
        return results
    
    def validate_audio_dataset(self, segments_path):
        """Детальна валідація аудіодатасету"""
        try:
            validator = AudioDatasetValidator(self.sr)
            return validator.validate_segments_structure(segments_path)
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Помилка валідації аудіо: {e}'
            }
    
    def validate_fft_dataset(self, fft_path):
        """Детальна валідація FFT датасету"""
        try:
            validator = FFTDatasetValidator()
            return validator.validate_fft_dataset(fft_path)
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Помилка валідації FFT: {e}'
            }
    
    def cross_validate_datasets(self, segments_path, fft_path):
        """Крос-валідація між аудіо та FFT датасетами"""
        try:
            cross_validator = CrossDatasetValidator()
            return cross_validator.validate_consistency(segments_path, fft_path)
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Помилка крос-валідації: {e}'
            }
    
    def generate_comprehensive_recommendations(self, results):
        """Генерація комплексних рекомендацій"""
        recommendations = []
        
        try:
            # Аналіз аудіодатасету
            audio_val = results.get('audio_validation', {})
            if audio_val.get('status') == 'success':
                success_rate = audio_val.get('success_rate', 0)
                if success_rate >= 95:
                    recommendations.append("Відмінна якість аудіодатасету!")
                elif success_rate >= 85:
                    recommendations.append("Хороша якість аудіодатасету")
                elif success_rate >= 70:
                    recommendations.append("Прийнятна якість аудіодатасету, потрібні незначні покращення")
                else:
                    recommendations.append("Низька якість аудіодатасету, потрібна переобробка")
            
            # Аналіз FFT датасету
            fft_val = results.get('fft_validation', {})
            if fft_val.get('status') == 'success':
                quality_score = fft_val.get('quality_score', 0)
                if quality_score >= 0.7:
                    recommendations.append("FFT датасет готовий для ML тренування")
                elif quality_score >= 0.6:
                    recommendations.append("FFT датасет потребує балансування класів")
                else:
                    recommendations.append("FFT датасет потребує значних покращень")
            
            # Крос-валідація
            cross_val = results.get('cross_validation', {})
            if cross_val.get('status') == 'success':
                consistency = cross_val.get('consistency_score', 0)
                if consistency >= 0.95:
                    recommendations.append("Відмінна консистентність між датасетами")
                elif consistency >= 0.85:
                    recommendations.append("Хороша консистентність, незначні розбіжності")
                else:
                    recommendations.append("Низька консистентність між датасетами")
            
        except Exception as e:
            recommendations.append(f"Помилка генерації рекомендацій: {e}")
        
        return recommendations
    
    def generate_comprehensive_report(self, results, base_path):
        """Генерація комплексного звіту"""
        report_path = os.path.join(base_path, 'comprehensive_validation_report.json')
        
        try:
            # Використовуємо безпечний енкодер
            safe_results = SafeJSONEncoder.encode_value(results)
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(safe_results, f, indent=2, ensure_ascii=False)
            
            text_report_path = os.path.join(base_path, 'validation_summary.txt')
            self.generate_text_report(results, text_report_path)
            
            print(f"\nЗвіти збережено:")
            print(f"   JSON: {report_path}")
            print(f"   Text: {text_report_path}")
            
        except Exception as e:
            print(f"Помилка збереження звіту: {e}")
            # Спробуємо зберегти спрощений звіт
            try:
                simple_report = {
                    'timestamp': results.get('timestamp'),
                    'base_path': results.get('base_path'),
                    'recommendations': results.get('recommendations', [])
                }
                simple_path = os.path.join(base_path, 'simple_validation_report.json')
                with open(simple_path, 'w', encoding='utf-8') as f:
                    json.dump(simple_report, f, indent=2, ensure_ascii=False)
                print(f"Спрощений звіт збережено: {simple_path}")
            except Exception as e2:
                print(f"Не вдалося зберегти навіть спрощений звіт: {e2}")
    
    def generate_text_report(self, results, output_path):
        """Генерація текстового звіту"""
        try:
            lines = []
            lines.append("КОМПЛЕКСНИЙ ЗВІТ ВАЛІДАЦІЇ ДАТАСЕТУ")
            lines.append("=" * 50)
            lines.append(f"Час генерації: {results.get('timestamp', 'N/A')}")
            lines.append(f"Базовий шлях: {results.get('base_path', 'N/A')}")
            lines.append("")
            
            # Аудіодатасет
            audio_val = results.get('audio_validation', {})
            if audio_val.get('status') == 'success':
                lines.append("АУДІОДАТАСЕТ:")
                lines.append(f"  Всього файлів: {audio_val.get('total_files', 0)}")
                lines.append(f"  Валідних файлів: {audio_val.get('valid_files', 0)}")
                lines.append(f"  Успішність: {audio_val.get('success_rate', 0):.1f}%")
                lines.append("")
            
            # FFT датасет
            fft_val = results.get('fft_validation', {})
            if fft_val.get('status') == 'success':
                lines.append("FFT ДАТАСЕТ:")
                lines.append(f"  Кількість зразків: {fft_val.get('total_samples', 0)}")
                lines.append(f"  Кількість класів: {fft_val.get('num_classes', 0)}")
                lines.append(f"  Якість: {fft_val.get('quality_score', 0):.3f}")
                lines.append("")
            
            # Рекомендації
            lines.append("РЕКОМЕНДАЦІЇ:")
            for rec in results.get('recommendations', []):
                lines.append(f"  {rec}")
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
                
        except Exception as e:
            print(f"Помилка генерації текстового звіту: {e}")
    
    def print_final_summary(self, results):
        """Виведення фінального резюме"""
        print(f"\n{'='*60}")
        print("ФІНАЛЬНЕ РЕЗЮМЕ")
        print(f"{'='*60}")
        
        audio_status = results.get('audio_validation', {}).get('status', 'unknown')
        fft_status = results.get('fft_validation', {}).get('status', 'unknown')
        cross_status = results.get('cross_validation', {}).get('status', 'unknown')
        
        print(f"Аудіодатасет: {self.get_status_emoji(audio_status)} {audio_status}")
        print(f"FFT датасет: {self.get_status_emoji(fft_status)} {fft_status}")
        print(f"Крос-валідація: {self.get_status_emoji(cross_status)} {cross_status}")
        
        print(f"\nКЛЮЧОВІ РЕКОМЕНДАЦІЇ:")
        for rec in results.get('recommendations', []):
            print(f"   {rec}")
    
    def get_status_emoji(self, status):
        """Отримання емодзі для статусу"""
        emoji_map = {
            'success': '✅',
            'warning': '⚠️',
            'error': '❌',
            'not_found': '📂',
            'unknown': '❓'
        }
        return emoji_map.get(status, '❓')


class AudioDatasetValidator:
    """Валідатор аудіодатасету"""
    
    def __init__(self, sr=44100):
        self.sr = sr
        self.stats = {
            'total_files': 0,
            'valid_files': 0,
            'errors': 0
        }
        self.quality_metrics = {
            'rms_db': [],
            'duration': [],
            'peak': []
        }
        
    def validate_segments_structure(self, segments_path):
        """Валідація структури сегментів"""
        print("Аналіз структури аудіосегментів...")
        
        try:
            structure_analysis = self.analyze_directory_structure(segments_path)
            audio_quality = self.analyze_audio_quality(segments_path)
            balance_analysis = self.analyze_class_balance(segments_path)
            
            success_rate = self.calculate_success_rate()
            
            return {
                'status': 'success' if success_rate >= 70 else 'warning',
                'success_rate': success_rate,
                'total_files': self.stats['total_files'],
                'valid_files': self.stats['valid_files'],
                'structure_analysis': structure_analysis,
                'audio_quality': audio_quality,
                'balance_analysis': balance_analysis,
                'detailed_stats': dict(self.stats)
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def analyze_directory_structure(self, segments_path):
        """Аналіз структури директорій"""
        expected_variants = ['original', 'up_semitone', 'down_semitone']
        found_variants = []
        
        for variant in expected_variants:
            variant_path = os.path.join(segments_path, variant)
            if os.path.exists(variant_path):
                found_variants.append(variant)
        
        structure_score = len(found_variants) / len(expected_variants)
        
        return {
            'expected_variants': expected_variants,
            'found_variants': found_variants,
            'structure_complete': structure_score == 1.0,
            'structure_score': structure_score
        }
    
    def analyze_audio_quality(self, segments_path):
        """Аналіз якості аудіо"""
        quality_issues = defaultdict(int)
        
        for root, dirs, files in os.walk(segments_path):
            for file in files:
                if file.endswith('.wav'):
                    file_path = os.path.join(root, file)
                    issues = self.check_single_audio_file(file_path)
                    
                    for issue in issues:
                        quality_issues[issue] += 1
        
        return {
            'quality_issues': dict(quality_issues),
            'average_rms': float(np.mean(self.quality_metrics['rms_db'])) if self.quality_metrics['rms_db'] else 0,
            'rms_std': float(np.std(self.quality_metrics['rms_db'])) if self.quality_metrics['rms_db'] else 0,
            'duration_consistency': self.check_duration_consistency()
        }
    
    def check_single_audio_file(self, file_path):
        """Перевірка одного аудіофайлу"""
        issues = []
        self.stats['total_files'] += 1
        
        try:
            audio, sr = librosa.load(file_path, sr=self.sr, mono=True)
            
            # Базові перевірки
            duration = len(audio) / sr
            rms = np.sqrt(np.mean(audio**2))
            peak = np.max(np.abs(audio))
            
            if rms > 0:
                rms_db = 20 * np.log10(rms)
            else:
                rms_db = -100
                issues.append('silent_file')
            
            self.quality_metrics['rms_db'].append(rms_db)
            self.quality_metrics['duration'].append(duration)
            self.quality_metrics['peak'].append(peak)
            
            # Перевірка тривалості
            if duration < 1.8:
                issues.append('too_short')
            elif duration > 3.0:
                issues.append('too_long')
            
            # Перевірка рівня
            if rms_db < -60:
                issues.append('too_quiet')
            elif rms_db > -6:
                issues.append('too_loud')
            
            # Перевірка кліпування
            if np.sum(np.abs(audio) > 0.98) / len(audio) > 0.01:
                issues.append('clipping')
            
            # Перевірка тиші
            silence_threshold = max(peak * 0.02, rms * 0.1)
            if np.sum(np.abs(audio) < silence_threshold) / len(audio) > 0.8:
                issues.append('too_much_silence')
            
            if len(issues) == 0:
                self.stats['valid_files'] += 1
            
            return issues
            
        except Exception as e:
            issues.append('load_error')
            self.stats['errors'] += 1
            return issues
    
    def analyze_class_balance(self, segments_path):
        """Аналіз балансу класів"""
        class_counts = defaultdict(int)
        
        for root, dirs, files in os.walk(segments_path):
            # Витягуємо ім'я інтервалу з шляху
            path_parts = root.split(os.sep)
            if len(path_parts) >= 2:
                interval_name = path_parts[-2]  # Передостання частина шляху
                wav_files = [f for f in files if f.endswith('.wav')]
                class_counts[interval_name] += len(wav_files)
        
        if class_counts:
            counts = list(class_counts.values())
            balance_score = min(counts) / max(counts) if max(counts) > 0 else 0
            cv = np.std(counts) / np.mean(counts) if np.mean(counts) > 0 else float('inf')
        else:
            balance_score = 0
            cv = float('inf')
        
        return {
            'class_counts': dict(class_counts),
            'balance_score': balance_score,
            'coefficient_of_variation': cv,
            'is_balanced': balance_score >= 0.7 and cv <= 0.3
        }
    
    def check_duration_consistency(self):
        """Перевірка консистентності тривалості"""
        if not self.quality_metrics['duration']:
            return {'consistent': False, 'std': 0}
        
        durations = self.quality_metrics['duration']
        std_duration = np.std(durations)
        mean_duration = np.mean(durations)
        
        # Консистентним вважається, якщо стандартне відхилення < 10% від середнього
        is_consistent = (std_duration / mean_duration) < 0.1 if mean_duration > 0 else False
        
        return {
            'consistent': is_consistent,
            'std': float(std_duration),
            'mean': float(mean_duration),
            'coefficient_of_variation': float(std_duration / mean_duration) if mean_duration > 0 else float('inf')
        }
    
    def calculate_success_rate(self):
        """Розрахунок загального показника успішності"""
        if self.stats['total_files'] == 0:
            return 0
        return (self.stats['valid_files'] / self.stats['total_files']) * 100


class FFTDatasetValidator:
    """Валідатор FFT датасету"""
    
    def __init__(self):
        self.quality_metrics = {}
        
    def validate_fft_dataset(self, fft_path):
        """Валідація FFT датасету"""
        print("Аналіз FFT датасету...")
        
        # Пошук файлів датасету
        pickle_path = os.path.join(fft_path, 'fft_dataset.pkl')
        npz_path = os.path.join(fft_path, 'fft_dataset.npz')
        metadata_path = os.path.join(fft_path, 'fft_dataset_metadata.json')
        
        if os.path.exists(pickle_path):
            return self.validate_pickle_dataset(pickle_path, metadata_path)
        elif os.path.exists(npz_path):
            return self.validate_npz_dataset(npz_path, metadata_path)
        else:
            return {
                'status': 'error',
                'message': 'FFT датасет не знайдено'
            }
    
    def validate_pickle_dataset(self, pickle_path, metadata_path):
        """Валідація pickle датасету"""
        try:
            with open(pickle_path, 'rb') as f:
                dataset = pickle.load(f)
            
            X = dataset['features']
            y = dataset['labels']
            
            return self.analyze_fft_data(X, y, dataset.get('metadata', {}))
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Помилка завантаження pickle датасету: {e}'
            }
    
    def validate_npz_dataset(self, npz_path, metadata_path):
        """Валідація NPZ датасету"""
        try:
            data = np.load(npz_path)
            X = data['features']
            y = data['labels']
            
            metadata = {}
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            
            return self.analyze_fft_data(X, y, metadata.get('metadata', {}))
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Помилка завантаження NPZ датасету: {e}'
            }
    
    def analyze_fft_data(self, X, y, metadata):
        """Аналіз FFT даних"""
        analysis = {
            'status': 'success',
            'total_samples': int(len(X)),
            'feature_dimension': int(X.shape[1]),
            'num_classes': int(len(np.unique(y))),
            'class_names': [str(cls) for cls in np.unique(y)],
            'metadata': metadata
        }
        
        # Аналіз якості ознак
        feature_quality = self.analyze_feature_quality(X)
        analysis['feature_quality'] = feature_quality
        
        # Аналіз балансу класів
        class_balance = self.analyze_fft_class_balance(y)
        analysis['class_balance'] = class_balance
        
        # Аналіз сепарабельності (тільки якщо sklearn доступний)
        if SKLEARN_AVAILABLE and len(X) > 100:
            separability = self.analyze_class_separability(X, y)
            analysis['separability'] = separability
        
        # Розрахунок загальної якості
        quality_score = self.calculate_fft_quality_score(analysis)
        analysis['quality_score'] = quality_score
        
        return analysis
    
    def analyze_feature_quality(self, X):
        """Аналіз якості ознак"""
        # Перевірка NaN та Inf значень
        nan_count = int(np.sum(np.isnan(X)))
        inf_count = int(np.sum(np.isinf(X)))
        
        # Аналіз варіації ознак
        feature_vars = np.var(X, axis=0)
        low_variance_features = int(np.sum(feature_vars < 1e-8))
        
        # Аналіз кореляції
        try:
            feature_corr = np.corrcoef(X.T)
            high_corr_pairs = int(np.sum(np.abs(feature_corr) > 0.95) - X.shape[1])  # Виключаємо діагональ
        except:
            high_corr_pairs = 0
        
        return {
            'nan_values': nan_count,
            'inf_values': inf_count,
            'low_variance_features': low_variance_features,
            'high_correlation_pairs': high_corr_pairs // 2,  # Кожна пара рахується двічі
            'feature_ranges': {
                'min': float(np.min(X)),
                'max': float(np.max(X)),
                'mean': float(np.mean(X)),
                'std': float(np.std(X))
            }
        }
    
    def analyze_fft_class_balance(self, y):
        """Аналіз балансу класів у FFT датасеті"""
        class_counts = Counter(y)
        counts = list(class_counts.values())
        
        balance_score = min(counts) / max(counts) if max(counts) > 0 else 0
        cv = np.std(counts) / np.mean(counts) if np.mean(counts) > 0 else float('inf')
        
        return {
            'class_counts': {str(k): int(v) for k, v in class_counts.items()},
            'balance_score': float(balance_score),
            'coefficient_of_variation': float(cv),
            'is_balanced': balance_score >= 0.9,
            'total_samples': int(np.sum(counts)),
            'classes': len(class_counts)
        }
    
    def analyze_class_separability(self, X, y):
        """Аналіз сепарабельності класів"""
        if not SKLEARN_AVAILABLE:
            return {'error': 'sklearn недоступний', 'separable': False}
        
        try:
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            
            if len(X) > 1000:
                indices = np.random.choice(len(X), 1000, replace=False)
                X_sample = X[indices]
                y_sample = y_encoded[indices]
            else:
                X_sample = X
                y_sample = y_encoded
            
            rf = RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=1)
            scores = cross_val_score(rf, X_sample, y_sample, cv=3, scoring='accuracy')
            
            separability_score = float(np.mean(scores))
            
            return {
                'cross_val_accuracy': separability_score,
                'cross_val_std': float(np.std(scores)),
                'separable': separability_score >= 0.8
            }
            
        except Exception as e:
            return {
                'error': f'Помилка аналізу сепарабельності: {e}',
                'separable': False
            }
    
    def calculate_fft_quality_score(self, analysis):
        """Розрахунок загального показника якості FFT датасету"""
        score = 0.0
        
        # Базові критерії (40%)
        if analysis['total_samples'] >= 1000:
            score += 0.2
        elif analysis['total_samples'] >= 500:
            score += 0.1
        
        if analysis['num_classes'] >= 8:
            score += 0.2
        elif analysis['num_classes'] >= 5:
            score += 0.1
        
        # Якість ознак (30%)
        feature_quality = analysis.get('feature_quality', {})
        if feature_quality.get('nan_values', 0) == 0:
            score += 0.1
        if feature_quality.get('inf_values', 0) == 0:
            score += 0.1
        if feature_quality.get('low_variance_features', 0) < analysis['feature_dimension'] * 0.1:
            score += 0.1
        
        # Баланс класів (20%)
        class_balance = analysis.get('class_balance', {})
        if class_balance.get('is_balanced', False):
            score += 0.2
        elif class_balance.get('balance_score', 0) >= 0.5:
            score += 0.1
        
        # Сепарабельність (10%)
        separability = analysis.get('separability', {})
        if separability.get('separable', False):
            score += 0.1
        elif separability.get('cross_val_accuracy', 0) >= 0.6:
            score += 0.05
        
        return min(score, 1.0)


class CrossDatasetValidator:
    """Крос-валідатор між датасетами"""
    
    def validate_consistency(self, segments_path, fft_path):
        """Валідація консистентності між аудіо та FFT датасетами"""
        print("Крос-валідація датасетів...")
        
        try:
            # Підрахунок файлів в аудіодатасеті
            audio_counts = self.count_audio_files(segments_path)
            
            # Аналіз FFT датасету
            fft_analysis = self.analyze_fft_samples(fft_path)
            
            # Порівняння
            consistency_analysis = self.compare_datasets(audio_counts, fft_analysis)
            
            return {
                'status': 'success',
                'audio_counts': audio_counts,
                'fft_analysis': fft_analysis,
                'consistency_analysis': consistency_analysis,
                'consistency_score': consistency_analysis.get('consistency_score', 0)
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Помилка крос-валідації: {e}'
            }
    
    def count_audio_files(self, segments_path):
        """Підрахунок аудіофайлів по класах"""
        counts = defaultdict(int)
        total = 0
        
        for root, dirs, files in os.walk(segments_path):
            wav_files = [f for f in files if f.endswith('.wav')]
            if wav_files:
                # Витягуємо ім'я класу з шляху
                path_parts = root.split(os.sep)
                if len(path_parts) >= 2:
                    class_name = path_parts[-2]
                    counts[class_name] += len(wav_files)
                    total += len(wav_files)
        
        return {
            'by_class': dict(counts),
            'total': total,
            'classes': len(counts)
        }
    
    def analyze_fft_samples(self, fft_path):
        """Аналіз зразків у FFT датасеті"""
        pickle_path = os.path.join(fft_path, 'fft_dataset.pkl')
        
        if os.path.exists(pickle_path):
            try:
                with open(pickle_path, 'rb') as f:
                    dataset = pickle.load(f)
                
                y = dataset['labels']
                class_counts = Counter(y)
                
                return {
                    'by_class': {str(k): int(v) for k, v in class_counts.items()},
                    'total': len(y),
                    'classes': len(class_counts)
                }
            except Exception as e:
                return {'error': f'Помилка читання FFT датасету: {e}'}
        else:
            return {'error': 'FFT датасет не знайдено'}
    
    def compare_datasets(self, audio_counts, fft_analysis):
        """Порівняння датасетів"""
        if 'error' in fft_analysis:
            return {'error': 'Неможливо порівняти через відсутність FFT датасету'}
        
        audio_total = audio_counts.get('total', 0)
        fft_total = fft_analysis.get('total', 0)
        
        # Розрахунок співвідношення
        if audio_total > 0:
            conversion_ratio = fft_total / audio_total
        else:
            conversion_ratio = 0
        
        # Порівняння кількості класів
        audio_classes = set(audio_counts.get('by_class', {}).keys())
        fft_classes = set(fft_analysis.get('by_class', {}).keys())
        
        common_classes = audio_classes.intersection(fft_classes)
        missing_in_fft = audio_classes - fft_classes
        extra_in_fft = fft_classes - audio_classes
        
        # Розрахунок консистентності
        class_consistency = len(common_classes) / max(len(audio_classes), 1)
        
        # Загальна консистентність
        count_consistency = 1.0 - abs(1.0 - conversion_ratio) if conversion_ratio <= 2.0 else 0.5
        consistency_score = (class_consistency + count_consistency) / 2
        
        return {
            'audio_total': audio_total,
            'fft_total': fft_total,
            'conversion_ratio': float(conversion_ratio),
            'class_consistency': float(class_consistency),
            'common_classes': list(common_classes),
            'missing_in_fft': list(missing_in_fft),
            'extra_in_fft': list(extra_in_fft),
            'consistency_score': float(consistency_score),
            'is_consistent': consistency_score >= 0.85
        }


class DatasetQualityReporter:
    """Генератор детальних звітів про якість датасету"""
    
    def __init__(self):
        self.report_data = {}
    
    def generate_detailed_report(self, validation_results, output_path):
        """Генерація детального звіту з візуалізацією"""
        
        # Створення директорії для звітів
        report_dir = os.path.join(output_path, 'quality_reports')
        Path(report_dir).mkdir(parents=True, exist_ok=True)
        
        # HTML звіт
        html_report = self.create_html_report(validation_results)
        html_path = os.path.join(report_dir, 'dataset_quality_report.html')
        
        try:
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_report)
        except Exception as e:
            print(f"Помилка створення HTML звіту: {e}")
            return None
        
        # Графіки якості (якщо matplotlib доступний)
        if PLOTTING_AVAILABLE:
            try:
                if validation_results.get('audio_validation', {}).get('status') == 'success':
                    self.create_audio_quality_plots(validation_results['audio_validation'], report_dir)
                
                if validation_results.get('fft_validation', {}).get('status') == 'success':
                    self.create_fft_quality_plots(validation_results['fft_validation'], report_dir)
            except Exception as e:
                print(f"Помилка створення графіків: {e}")
        
        print(f"Детальний звіт створено: {html_path}")
        return html_path
    
    def create_html_report(self, results):
        """Створення HTML звіту"""
        html = f"""
<!DOCTYPE html>
<html lang="uk">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Звіт якості датасету</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }}
        .header {{ text-align: center; color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
        .section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #4CAF50; background: #f9f9f9; }}
        .metric {{ display: inline-block; margin: 10px; padding: 10px; background: #e8f5e8; border-radius: 5px; }}
        .warning {{ border-left-color: #ff9800; background: #fff3e0; }}
        .error {{ border-left-color: #f44336; background: #ffebee; }}
        .success {{ border-left-color: #4CAF50; background: #e8f5e8; }}
        table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
        th, td {{ padding: 8px; text-align: left; border: 1px solid #ddd; }}
        th {{ background-color: #4CAF50; color: white; }}
        .recommendation {{ background: #e3f2fd; padding: 10px; margin: 5px 0; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Звіт якості датасету</h1>
            <p>Згенеровано: {results.get('timestamp', 'N/A')}</p>
        </div>
"""
        
        # Додаємо секції для кожного типу валідації
        if 'audio_validation' in results:
            html += self.create_audio_section(results['audio_validation'])
        
        if 'fft_validation' in results:
            html += self.create_fft_section(results['fft_validation'])
        
        if 'cross_validation' in results:
            html += self.create_cross_validation_section(results['cross_validation'])
        
        # Рекомендації
        html += f"""
        <div class="section">
            <h2>Рекомендації</h2>
            {"".join(f'<div class="recommendation">{rec}</div>' for rec in results.get('recommendations', []))}
        </div>
    </div>
</body>
</html>"""
        
        return html
    
    def create_audio_section(self, audio_results):
        """Створення секції аудіо валідації"""
        if audio_results.get('status') == 'error':
            return f"""
        <div class="section error">
            <h2>Аудіодатасет</h2>
            <p>{audio_results.get('message', 'Помилка валідації')}</p>
        </div>"""
        
        status_class = 'success' if audio_results.get('success_rate', 0) >= 85 else 'warning' if audio_results.get('success_rate', 0) >= 70 else 'error'
        
        balance_data = audio_results.get('balance_analysis', {}).get('class_counts', {})
        balance_table = ""
        if balance_data:
            balance_table = f"""
            <h3>Баланс класів</h3>
            <table>
                <tr><th>Клас</th><th>Кількість файлів</th></tr>
                {"".join(f"<tr><td>{cls}</td><td>{count}</td></tr>" for cls, count in balance_data.items())}
            </table>"""
        
        return f"""
        <div class="section {status_class}">
            <h2>Аудіодатасет</h2>
            <div class="metric">Всього файлів: <strong>{audio_results.get('total_files', 0)}</strong></div>
            <div class="metric">Валідних файлів: <strong>{audio_results.get('valid_files', 0)}</strong></div>
            <div class="metric">Успішність: <strong>{audio_results.get('success_rate', 0):.1f}%</strong></div>
            
            <h3>Структура датасету</h3>
            <p>Знайдені варіанти: {', '.join(audio_results.get('structure_analysis', {}).get('found_variants', []))}</p>
            
            {balance_table}
        </div>"""
    
    def create_fft_section(self, fft_results):
        """Створення секції FFT валідації"""
        if fft_results.get('status') != 'success':
            return f"""
        <div class="section error">
            <h2>FFT Датасет</h2>
            <p>{fft_results.get('message', 'Помилка валідації')}</p>
        </div>"""
        
        quality_score = fft_results.get('quality_score', 0)
        status_class = 'success' if quality_score >= 0.8 else 'warning' if quality_score >= 0.6 else 'error'
        
        balance_data = fft_results.get('class_balance', {}).get('class_counts', {})
        balance_table = ""
        if balance_data:
            balance_table = f"""
            <h3>Баланс класів</h3>
            <table>
                <tr><th>Клас</th><th>Зразків</th></tr>
                {"".join(f"<tr><td>{cls}</td><td>{count}</td></tr>" for cls, count in balance_data.items())}
            </table>"""
        
        return f"""
        <div class="section {status_class}">
            <h2>FFT Датасет</h2>
            <div class="metric">Зразків: <strong>{fft_results.get('total_samples', 0)}</strong></div>
            <div class="metric">Класів: <strong>{fft_results.get('num_classes', 0)}</strong></div>
            <div class="metric">Ознак: <strong>{fft_results.get('feature_dimension', 0)}</strong></div>
            <div class="metric">Якість: <strong>{quality_score:.3f}</strong></div>
            
            {balance_table}
        </div>"""
    
    def create_cross_validation_section(self, cross_results):
        """Створення секції крос-валідації"""
        if cross_results.get('status') != 'success':
            return f"""
        <div class="section error">
            <h2>Крос-валідація</h2>
            <p>{cross_results.get('message', 'Помилка крос-валідації')}</p>
        </div>"""
        
        consistency = cross_results.get('consistency_score', 0)
        status_class = 'success' if consistency >= 0.9 else 'warning' if consistency >= 0.7 else 'error'
        
        analysis = cross_results.get('consistency_analysis', {})
        
        return f"""
        <div class="section {status_class}">
            <h2>Крос-валідація</h2>
            <div class="metric">Консистентність: <strong>{consistency:.3f}</strong></div>
            <div class="metric">Співвідношення конверсії: <strong>{analysis.get('conversion_ratio', 0):.2f}</strong></div>
            
            <h3>Порівняння датасетів</h3>
            <p>Аудіо файлів: {cross_results.get('audio_counts', {}).get('total', 0)}</p>
            <p>FFT зразків: {cross_results.get('fft_analysis', {}).get('total', 0)}</p>
        </div>"""
    
    def create_audio_quality_plots(self, audio_results, output_dir):
        """Створення графіків якості аудіо"""
        if not PLOTTING_AVAILABLE:
            return
        
        try:
            # Графік балансу класів
            class_counts = audio_results.get('balance_analysis', {}).get('class_counts', {})
            if class_counts:
                plt.figure(figsize=(12, 6))
                plt.bar(class_counts.keys(), class_counts.values())
                plt.title('Розподіл класів в аудіодатасеті')
                plt.xlabel('Клас')
                plt.ylabel('Кількість файлів')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'audio_class_distribution.png'), dpi=300, bbox_inches='tight')
                plt.close()
                
        except Exception as e:
            print(f"Помилка створення графіків аудіо: {e}")
    
    def create_fft_quality_plots(self, fft_results, output_dir):
        """Створення графіків якості FFT"""
        if not PLOTTING_AVAILABLE:
            return
        
        try:
            # Графік балансу класів FFT
            class_counts = fft_results.get('class_balance', {}).get('class_counts', {})
            if class_counts:
                plt.figure(figsize=(12, 6))
                plt.bar(class_counts.keys(), class_counts.values())
                plt.title('Розподіл класів в FFT датасеті')
                plt.xlabel('Клас')
                plt.ylabel('Кількість зразків')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'fft_class_distribution.png'), dpi=300, bbox_inches='tight')
                plt.close()
                
        except Exception as e:
            print(f"Помилка створення графіків FFT: {e}")


# Функції швидкого доступу
def validate_dataset_quality(base_path, generate_reports=True):
    """Основна функція валідації якості датасету"""
    validator = ComprehensiveDatasetValidator()
    return validator.validate_complete_pipeline(base_path, generate_reports)

# Функції швидкого доступу
def validate_all(base_path="dataset-preparation"):
    """Валідація всього пайплайну"""
    return validate_dataset_quality(base_path, generate_reports=True)

def create_detailed_report(base_path="dataset-preparation", results=None):
    """Створення детального звіту з візуалізацією"""
    if results is None:
        results = validate_dataset_quality(base_path, generate_reports=False)
    reporter = DatasetQualityReporter()
    return reporter.generate_detailed_report(results, base_path)


if __name__ == "__main__":
    import sys
    
    # Визначення базового шляху
    if len(sys.argv) > 1:
        base_path = sys.argv[1]
    else:
        base_path = "dataset-preparation"
    
    print("ЗАПУСК КОМПЛЕКСНОЇ ВАЛІДАЦІЇ")
    print(f"Базовий шлях: {base_path}")
    print("=" * 60)
    
    if not os.path.exists(base_path):
        print(f"Шлях не існує: {base_path}")
        print("Створіть папку або вкажіть правильний шлях як аргумент")
        sys.exit(1)
    
    # Запуск валідації
    try:
        results = validate_all(base_path)
        
        # Створення детального звіту (передаємо вже отримані результати)
        print(f"\nСтворення детального звіту...")
        create_detailed_report(base_path, results)
        
        print(f"\nВалідація завершена!")
        
    except KeyboardInterrupt:
        print(f"\nВалідація перервана користувачем")
    except Exception as e:
        print(f"\nКритична помилка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)