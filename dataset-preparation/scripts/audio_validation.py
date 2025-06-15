import librosa
import numpy as np
import os
import json
from pathlib import Path
from collections import defaultdict
import time

class AudioValidator:
    def __init__(self, sr=44100):
        self.sr = sr
        self.stats = {
            'total_files': 0,
            'valid_files': 0,
            'issues': defaultdict(int),
            'quality_stats': {
                'rms_values': [],
                'peak_values': [],
                'silence_ratios': [],
                'durations': []
            },
            'quiet_files': [],
            'problematic_files': []
        }
    
    def _safe_json_convert(self, obj):
        """Безпечна конвертація для JSON серіалізації"""
        if isinstance(obj, (np.floating, np.complexfloating)):
            return float(obj)
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, defaultdict):
            return dict(obj)
        elif isinstance(obj, dict):
            return {key: self._safe_json_convert(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._safe_json_convert(item) for item in obj]
        else:
            return obj
    
    def validate_file(self, file_path):
        """Розширена валідація файлу з детальним аналізом"""
        try:
            audio, sr = librosa.load(file_path, sr=self.sr, mono=True)
            
            issues = []
            file_info = {
                'path': file_path,
                'basename': os.path.basename(file_path)
            }
            
            # 1. Перевірка довжини
            duration = len(audio) / sr
            file_info['duration'] = float(duration)
            self.stats['quality_stats']['durations'].append(duration)
            
            if duration < 1.8:
                issues.append('too_short')
            elif duration > 3.0:
                issues.append('too_long')
            
            # 2. Аналіз амплітуди та RMS
            rms = np.sqrt(np.mean(audio**2))
            peak = np.max(np.abs(audio))
            
            if rms > 0:
                rms_db = 20 * np.log10(rms)
            else:
                rms_db = -100
                
            if peak > 0:
                peak_db = 20 * np.log10(peak)
            else:
                peak_db = -100
            
            file_info['rms_db'] = float(rms_db)
            file_info['peak_db'] = float(peak_db)
            
            self.stats['quality_stats']['rms_values'].append(rms_db)
            self.stats['quality_stats']['peak_values'].append(peak_db)
            
            # Класифікація за рівнем
            if rms_db < -60:
                issues.append('too_quiet')
                self.stats['problematic_files'].append({
                    'file': os.path.basename(file_path),
                    'issue': 'too_quiet',
                    'rms_db': float(rms_db)
                })
            elif rms_db < -35:
                issues.append('quiet_fixable')
                self.stats['quiet_files'].append(file_path)
            elif rms_db > -6:
                issues.append('too_loud')
            
            # 3. Перевірка кліпування
            clipping_ratio = np.sum(np.abs(audio) > 0.98) / len(audio)
            file_info['clipping_ratio'] = float(clipping_ratio)
            
            if clipping_ratio > 0.01:
                issues.append('clipping')
                self.stats['problematic_files'].append({
                    'file': os.path.basename(file_path),
                    'issue': 'clipping',
                    'clipping_ratio': float(clipping_ratio * 100)
                })
            
            # 4. Перевірка тиші (адаптивний поріг)
            adaptive_threshold = max(peak * 0.02, rms * 0.1)
            silence_ratio = np.sum(np.abs(audio) < adaptive_threshold) / len(audio)
            
            file_info['silence_ratio'] = float(silence_ratio)
            self.stats['quality_stats']['silence_ratios'].append(silence_ratio)
            
            if silence_ratio > 0.8:
                issues.append('too_much_silence')
                self.stats['problematic_files'].append({
                    'file': os.path.basename(file_path),
                    'issue': 'too_much_silence',
                    'silence_ratio': float(silence_ratio * 100)
                })
            
            # 5. Перевірка спектральних характеристик
            try:
                zcr = librosa.feature.zero_crossing_rate(audio)[0]
                mean_zcr = np.mean(zcr)
                file_info['zero_crossing_rate'] = float(mean_zcr)
                
                if mean_zcr < 0.01:
                    issues.append('monotone_signal')
                
                spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
                mean_centroid = np.mean(spectral_centroid)
                file_info['spectral_centroid'] = float(mean_centroid)
                
                if mean_centroid > 8000:
                    issues.append('high_frequency_artifacts')
                elif mean_centroid < 200:
                    issues.append('low_frequency_only')
                
            except Exception as e:
                issues.append('spectral_analysis_failed')
            
            # 6. Перевірка на дефектний файл
            if np.all(audio == 0):
                issues.append('silent_file')
            elif np.std(audio) < 1e-7:
                issues.append('constant_signal')
            
            # 7. Перевірка динамічного діапазону
            if peak > 0 and rms > 0:
                dynamic_range = peak_db - rms_db
                file_info['dynamic_range'] = float(dynamic_range)
                
                if dynamic_range < 3:
                    issues.append('low_dynamic_range')
                elif dynamic_range > 40:
                    issues.append('excessive_dynamic_range')
            
            return len(issues) == 0, issues, file_info
            
        except Exception as e:
            error_info = {
                'path': file_path,
                'basename': os.path.basename(file_path),
                'error': str(e)
            }
            return False, ['load_error'], error_info
    
    def validate_dataset(self, dataset_path, auto_fix_quiet=False):
        """Розширена валідація всього датасету"""
        print("ВАЛІДАЦІЯ ДАТАСЕТУ")
        print("=" * 50)
        
        start_time = time.time()
        
        all_files = []
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.endswith('.wav') and not file.endswith('_original.wav'):
                    all_files.append(os.path.join(root, file))
        
        self.stats['total_files'] = len(all_files)
        detailed_info = []
        
        if not all_files:
            print("Не знайдено WAV файлів для валідації!")
            return False
        
        print(f"Знайдено {len(all_files)} файлів для валідації")
        print("-" * 50)
        
        # Валідація
        for i, file_path in enumerate(all_files):
            if i % 200 == 0 and i > 0:
                elapsed = time.time() - start_time
                progress = i / len(all_files) * 100
                eta = (elapsed / i) * (len(all_files) - i)
                print(f"Прогрес: {progress:.1f}% | Перевірено: {i}/{len(all_files)} | ETA: {eta:.0f}с")
            
            is_valid, issues, file_info = self.validate_file(file_path)
            detailed_info.append({**file_info, 'issues': issues, 'is_valid': is_valid})
            
            if is_valid:
                self.stats['valid_files'] += 1
            else:
                for issue in issues:
                    self.stats['issues'][issue] += 1
        
        # Розрахунок результатів
        success_rate = (self.stats['valid_files'] / self.stats['total_files'] * 100) if self.stats['total_files'] > 0 else 0
        elapsed_total = time.time() - start_time
        
        print(f"\n{'='*50}")
        print("РЕЗУЛЬТАТИ ВАЛІДАЦІЇ")
        print(f"{'='*50}")
        print(f"Всього файлів: {self.stats['total_files']}")
        print(f"Валідних файлів: {self.stats['valid_files']}")
        print(f"Проблемних файлів: {self.stats['total_files'] - self.stats['valid_files']}")
        print(f"Успішність: {success_rate:.1f}%")
        print(f"Час валідації: {elapsed_total:.1f} секунд")
        
        # Статистика проблем
        if self.stats['issues']:
            print(f"\nДЕТАЛЬНА СТАТИСТИКА ПРОБЛЕМ:")
            issue_descriptions = {
                'too_short': 'Занадто короткі (< 1.8с)',
                'too_long': 'Занадто довгі (> 3.0с)',
                'too_quiet': 'Дуже тихі (< -60дБ)',
                'quiet_fixable': 'Тихі, можна виправити (-60дБ до -35дБ)',
                'too_loud': 'Занадто гучні (> -6дБ)',
                'clipping': 'Клипування (> 1%)',
                'too_much_silence': 'Надмірна тиша (> 85%)',
                'monotone_signal': 'Монотонний сигнал',
                'high_frequency_artifacts': 'Високочастотні артефакти',
                'low_frequency_only': 'Тільки низькі частоти',
                'spectral_analysis_failed': 'Помилка спектрального аналізу',
                'silent_file': 'Повністю тихий файл',
                'constant_signal': 'Константний сигнал',
                'low_dynamic_range': 'Низький динамічний діапазон',
                'excessive_dynamic_range': 'Надмірний динамічний діапазон',
                'load_error': 'Помилки завантаження'
            }
            
            for issue, count in sorted(self.stats['issues'].items(), key=lambda x: x[1], reverse=True):
                desc = issue_descriptions.get(issue, issue)
                percentage = (count / self.stats['total_files']) * 100
                print(f"  {desc}: {count} файлів ({percentage:.1f}%)")
        
        # Статистика якості
        self._print_quality_statistics()
        
        # Автокорекція тихих файлів
        if auto_fix_quiet and len(self.stats['quiet_files']) > 0:
            print(f"\nАВТОКОРЕКЦІЯ ТИХИХ ФАЙЛІВ:")
            print(f"Знайдено {len(self.stats['quiet_files'])} файлів для корекції")
            success = self._auto_fix_quiet_files()
            if success:
                print("Рекомендовано повтор валідації після корекції")
        
        self._save_comprehensive_report(dataset_path, detailed_info, success_rate)
        
        self._print_recommendations(success_rate)
        
        return success_rate >= 70
    
    def _print_quality_statistics(self):
        """Виведення статистики якості"""
        if not self.stats['quality_stats']['rms_values']:
            return
        
        rms_values = [x for x in self.stats['quality_stats']['rms_values'] if not np.isnan(x) and not np.isinf(x)]
        silence_ratios = self.stats['quality_stats']['silence_ratios']
        durations = self.stats['quality_stats']['durations']
        
        print(f"\nСТАТИСТИКА ЯКОСТІ:")
        
        if rms_values:
            print(f"RMS рівні (дБ):")
            print(f"  Мінімум: {min(rms_values):.1f}")
            print(f"  Максимум: {max(rms_values):.1f}")
            print(f"  Середнє: {np.mean(rms_values):.1f}")
            print(f"  Медіана: {np.median(rms_values):.1f}")
        
        if silence_ratios:
            print(f"Співвідношення тиші:")
            print(f"  Середнє: {np.mean(silence_ratios)*100:.1f}%")
            print(f"  Медіана: {np.median(silence_ratios)*100:.1f}%")
        
        if durations:
            print(f"Тривалість файлів:")
            print(f"  Середня: {np.mean(durations):.2f}с")
            print(f"  Діапазон: {min(durations):.2f}с - {max(durations):.2f}с")
    
    def _auto_fix_quiet_files(self):
        """Автоматична корекція тихих файлів"""
        try:
            from audio_gain_helper import AudioGainProcessor
            
            processor = AudioGainProcessor()
            processed_count = 0
            
            for file_path in self.stats['quiet_files']:
                try:
                    if processor.process_file(file_path, backup=True):
                        processed_count += 1
                except Exception as e:
                    print(f"  Помилка корекції {os.path.basename(file_path)}: {e}")
            
            print(f"Успішно оброблено: {processed_count}/{len(self.stats['quiet_files'])} файлів")
            return processed_count > 0
            
        except ImportError:
            print("  ПОМИЛКА: Модуль audio_gain_helper не знайдено")
            return False
        except Exception as e:
            print(f"  Помилка автокорекції: {e}")
            return False
    
    def _save_comprehensive_report(self, dataset_path, detailed_info, success_rate):
        """Збереження комплексного звіту"""
        report_path = os.path.join(dataset_path, 'validation_report.json')
        
        try:
            report_data = {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'dataset_path': dataset_path,
                'summary': {
                    'total_files': int(self.stats['total_files']),
                    'valid_files': int(self.stats['valid_files']),
                    'problematic_files': int(self.stats['total_files'] - self.stats['valid_files']),
                    'success_rate': float(success_rate)
                },
                'issues_breakdown': self._safe_json_convert(dict(self.stats['issues'])),
                'quality_statistics': self._calculate_quality_stats(),
                'problematic_files_details': self.stats['problematic_files'][:20],
                'recommendations': self._generate_recommendations(success_rate),
                'sample_files': detailed_info[:10]
            }
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(self._safe_json_convert(report_data), f, indent=2, ensure_ascii=False)
            
            print(f"\nКомплексний звіт збережено: {report_path}")
            
        except Exception as e:
            print(f"Помилка збереження звіту: {e}")
            try:
                simple_report = {
                    'total_files': int(self.stats['total_files']),
                    'valid_files': int(self.stats['valid_files']),
                    'success_rate': float(success_rate),
                    'issues': self._safe_json_convert(dict(self.stats['issues']))
                }
                
                simple_path = os.path.join(dataset_path, 'validation_simple.json')
                with open(simple_path, 'w', encoding='utf-8') as f:
                    json.dump(simple_report, f, indent=2, ensure_ascii=False)
                print(f"Спрощений звіт збережено: {simple_path}")
                
            except Exception:
                print("Не вдалося зберегти спрощений звіт")
    
    def _calculate_quality_stats(self):
        """Розрахунок статистики якості"""
        stats = {}
        
        if self.stats['quality_stats']['rms_values']:
            rms_clean = [x for x in self.stats['quality_stats']['rms_values'] 
                        if not np.isnan(x) and not np.isinf(x)]
            if rms_clean:
                stats['rms_db'] = {
                    'min': float(min(rms_clean)),
                    'max': float(max(rms_clean)),
                    'mean': float(np.mean(rms_clean)),
                    'median': float(np.median(rms_clean)),
                    'std': float(np.std(rms_clean))
                }
        
        if self.stats['quality_stats']['silence_ratios']:
            silence = self.stats['quality_stats']['silence_ratios']
            stats['silence_ratio'] = {
                'mean': float(np.mean(silence)),
                'median': float(np.median(silence)),
                'max': float(max(silence))
            }
        
        if self.stats['quality_stats']['durations']:
            durations = self.stats['quality_stats']['durations']
            stats['duration'] = {
                'mean': float(np.mean(durations)),
                'min': float(min(durations)),
                'max': float(max(durations))
            }
        
        return stats
    
    def _generate_recommendations(self, success_rate):
        """Генерація рекомендацій для покращення датасету"""
        recommendations = []
        
        total_files = self.stats['total_files']
        
        silence_issues = self.stats['issues'].get('too_much_silence', 0)
        if silence_issues > total_files * 0.3:
            recommendations.append(
                "КРИТИЧНО: Багато файлів з надмірною тишею."
            )
        
        quiet_issues = self.stats['issues'].get('quiet_fixable', 0)
        if quiet_issues > 0:
            recommendations.append(
                f"Рекомендовано автокорекцію {quiet_issues} тихих файлів за допомогою audio_gain_helper.py"
            )
        
        clipping_issues = self.stats['issues'].get('clipping', 0)
        if clipping_issues > total_files * 0.1:
            recommendations.append(
                "Багато файлів з кліпуванням."
            )
        
        if success_rate >= 90:
            recommendations.append("Відмінна якість датасету! Готовий для тренування CNN.")
        elif success_rate >= 80:
            recommendations.append("Хороша якість датасету. Можна продовжувати з генерацією спектрограм.")
        elif success_rate >= 70:
            recommendations.append("Прийнятна якість. Рекомендуємо незначну доробку.")
        elif success_rate >= 50:
            recommendations.append("Середня якість. Потрібна серйозна переобробка аудіо.")
        else:
            recommendations.append("Низька якість датасету. Критично потрібна переобробка.")
        
        return recommendations
    
    def _print_recommendations(self, success_rate):
        """Виведення рекомендацій"""
        recommendations = self._generate_recommendations(success_rate)
        
        print(f"\nРЕКОМЕНДАЦІЇ:")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")


def validate_dataset(dataset_path, auto_fix_quiet=False):
    """Функція валідації"""
    validator = AudioValidator()
    return validator.validate_dataset(dataset_path, auto_fix_quiet)


def validate_and_fix(dataset_path):
    """Валідація з автоматичною корекцією тихих файлів"""
    print("ВАЛІДАЦІЯ З АВТОКОРЕКЦІЄЮ")
    print("=" * 60)
    
    validator1 = AudioValidator()
    result1 = validator1.validate_dataset(dataset_path, auto_fix_quiet=True)
    
    if len(validator1.stats['quiet_files']) > 0:
        print(f"\n{'='*60}")
        print("ПОВТОРНА ВАЛІДАЦІЯ ПІСЛЯ АВТОКОРЕКЦІЇ")
        print(f"{'='*60}")
        
        validator2 = AudioValidator()
        result2 = validator2.validate_dataset(dataset_path, auto_fix_quiet=False)
        
        print(f"\nПОРІВНЯННЯ РЕЗУЛЬТАТІВ:")
        print(f"До корекції:    {validator1.stats['valid_files']}/{validator1.stats['total_files']} "
              f"({validator1.stats['valid_files']/validator1.stats['total_files']*100:.1f}%)")
        print(f"Після корекції: {validator2.stats['valid_files']}/{validator2.stats['total_files']} "
              f"({validator2.stats['valid_files']/validator2.stats['total_files']*100:.1f}%)")
        
        improvement = validator2.stats['valid_files'] - validator1.stats['valid_files']
        if improvement > 0:
            print(f"Покращення: +{improvement} файлів")
        
        return result2
    
    return result1


if __name__ == "__main__":
    dataset_path = "dataset-preparation/processed-segments"
    
    if os.path.exists(dataset_path):
        validate_and_fix(dataset_path)
    else:
        print(f"Датасет не знайдено: {dataset_path}")
        print("Спочатку запустіть audio_processing.py для створення датасету")