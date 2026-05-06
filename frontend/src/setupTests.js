import '@testing-library/jest-dom';
import { vi } from 'vitest';

globalThis.Element.prototype.scrollIntoView = vi.fn();
globalThis.Element.prototype.scrollTo = vi.fn();
globalThis.window.scroll = vi.fn();
globalThis.window.scrollTo = vi.fn();
