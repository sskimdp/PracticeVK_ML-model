// точка входа приложения
import './styles/style.css';
import { App } from './ui/App.ts';

const root = document.querySelector<HTMLDivElement>('#app')!;
new App(root);
