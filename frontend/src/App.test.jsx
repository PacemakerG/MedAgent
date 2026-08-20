import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import App from './App';

function jsonResponse(payload, ok = true) {
    return Promise.resolve({
        json: () => Promise.resolve(payload),
        ok,
    });
}

function sseResponse(frames) {
    const encoder = new TextEncoder();
    return Promise.resolve({
        ok: true,
        body: new ReadableStream({
            start(controller) {
                frames.forEach(frame => controller.enqueue(encoder.encode(frame)));
                controller.close();
            },
        }),
    });
}

async function login() {
    fireEvent.change(await screen.findByLabelText('用户ID'), {
        target: { value: 'doctor_zhang' },
    });
    fireEvent.change(screen.getByLabelText('密码'), {
        target: { value: 'strong-password-123' },
    });
    fireEvent.click(screen.getByRole('button', { name: '登录并进入系统' }));

    await waitFor(() => {
        expect(screen.queryByText('登录 医枢智疗')).not.toBeInTheDocument();
    });
}

describe('App Integration', () => {
    const mockFetch = vi.fn();
    globalThis.fetch = mockFetch;

    beforeEach(() => {
        localStorage.clear();
        mockFetch.mockClear();
        mockFetch.mockImplementation((url) => {
            if (url === '/api/v1/auth/me') {
                return jsonResponse({
                    success: true,
                    logged_in: false,
                    user_id: 'anonymous',
                    session_id: 'session-1',
                });
            }
            if (url === '/api/v1/auth/login') {
                return jsonResponse({
                    success: true,
                    logged_in: true,
                    user_id: 'doctor_zhang',
                    session_id: 'session-1',
                    access_token: 'test-access-token',
                    token_type: 'Bearer',
                    expires_at: 9999999999,
                });
            }
            if (url === '/api/v1/sessions') {
                return jsonResponse({ success: true, sessions: [] });
            }
            if (url === '/api/v1/history') {
                return jsonResponse({
                    success: true,
                    session_id: 'session-1',
                    messages: [],
                });
            }
            if (url === '/api/v1/new-chat') {
                return jsonResponse({ success: true, session_id: 'new-session' });
            }
            if (url === '/api/v1/chat/stream') {
                return sseResponse([
                    'event: delta\ndata: {"delta":"I can help"}\n\n',
                    'event: delta\ndata: {"delta":" with that."}\n\n',
                    'event: done\ndata: {"success":true,"response":"I can help with that.","source":"test-source","timestamp":"now"}\n\n',
                ]);
            }
            return jsonResponse({ success: true });
        });
    });

    afterEach(() => {
        cleanup();
        vi.restoreAllMocks();
    });

    it('renders the password login gate', async () => {
        render(<App />);

        expect(await screen.findByText('登录 医枢智疗')).toBeInTheDocument();
        expect(screen.getByLabelText('用户ID')).toBeInTheDocument();
        expect(screen.getByLabelText('密码')).toBeInTheDocument();
        expect(screen.getByText('医枢智疗·临床协同中枢')).toBeInTheDocument();
    });

    it('loads sessions after login', async () => {
        mockFetch.mockImplementation((url) => {
            if (url === '/api/v1/auth/me') {
                return jsonResponse({
                    success: true,
                    logged_in: false,
                    user_id: 'anonymous',
                    session_id: 'session-1',
                });
            }
            if (url === '/api/v1/auth/login') {
                return jsonResponse({
                    success: true,
                    logged_in: true,
                    user_id: 'doctor_zhang',
                    session_id: 'session-1',
                    access_token: 'test-access-token',
                });
            }
            if (url === '/api/v1/sessions') {
                return jsonResponse({
                    success: true,
                    sessions: [{ session_id: '1', preview: 'Flu symptoms', last_active: '2023' }],
                });
            }
            if (url === '/api/v1/history') {
                return jsonResponse({ success: true, session_id: 'session-1', messages: [] });
            }
            return jsonResponse({ success: true });
        });

        render(<App />);
        await login();

        expect(await screen.findByText('Flu symptoms')).toBeInTheDocument();
    });

    it('sends a streamed message and displays response', async () => {
        render(<App />);
        await login();

        const input = screen.getByPlaceholderText('请输入你的问题（可直接描述症状、病史或检查结果）');
        fireEvent.change(input, { target: { value: 'Headache' } });
        fireEvent.click(screen.getByRole('button', { name: '发送消息' }));

        expect(screen.getByText('Headache')).toBeInTheDocument();
        expect(await screen.findByText('I can help with that.')).toBeInTheDocument();
    });

    it('creates new chat after login', async () => {
        render(<App />);
        await login();

        fireEvent.click(screen.getByText('新建会话'));

        await waitFor(() => {
            expect(mockFetch).toHaveBeenCalledWith('/api/v1/new-chat', expect.any(Object));
        });
    });

    it('shows welcome workspace after login', async () => {
        render(<App />);
        await login();

        expect(screen.getByText('欢迎使用 医枢智疗')).toBeInTheDocument();
        expect(screen.getByText('面向多科室问诊与 ECG 报告的一体化医疗智能工作台')).toBeInTheDocument();
    });
});
