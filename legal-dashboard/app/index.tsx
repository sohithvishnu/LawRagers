import React, { useState, useEffect, useRef } from 'react';
import { StyleSheet, Text, View, TextInput, TouchableOpacity, ScrollView, ActivityIndicator, KeyboardAvoidingView, Platform } from 'react-native';
import Svg, { Line } from 'react-native-svg';
import Markdown from 'react-native-markdown-display';
import { Ionicons } from '@expo/vector-icons';
import * as DocumentPicker from 'expo-document-picker';
import type { DocumentPickerAsset } from 'expo-document-picker';

const API_BASE = 'http://localhost:8000';

type Session = { id: string; name: string; description: string; databases: string; created_at: string };
type Message = { id: string; role: string; text: string; files?: string[]; graph_state?: string };
type Case = { id: string; date: string; text: string; distance: number; hitCount?: number };

export default function Index() {
  const [currentScreen, setCurrentScreen] = useState('lobby');

  const [savedSessions, setSavedSessions] = useState<Session[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);

  const [configName, setConfigName] = useState('');
  const [configDesc, setConfigDesc] = useState('');
  const [configDbNy, setConfigDbNy] = useState(true);
  const [configError, setConfigError] = useState('');

  const [sessionName, setSessionName] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [argument, setArgument] = useState('');

  const [uploadedFiles, setUploadedFiles] = useState<DocumentPickerAsset[]>([]);
  const [indexedFiles, setIndexedFiles] = useState<string[]>([]);

  const [loadingState, setLoadingState] = useState('');
  const [cases, setCases] = useState<Case[]>([]);
  const [selectedCase, setSelectedCase] = useState<Case | null>(null);
  const [zoomScale, setZoomScale] = useState(1);
  const scrollViewRef = useRef<ScrollView>(null);

  useEffect(() => {
    if (currentScreen === 'lobby') fetchSessions();
  }, [currentScreen]);

  const fetchSessions = async () => {
    try {
      const res = await fetch(`${API_BASE}/sessions`);
      const data = await res.json();
      setSavedSessions(data.sessions);
    } catch (error) { console.error("Failed to load sessions", error); }
  };

  const removeFile = (index: number) => {
    setUploadedFiles(prev => prev.filter((_, i) => i !== index));
  };

  // 1. UPLOAD HANDLERS
  const pickFile = async () => {
    try {
      const result = await DocumentPicker.getDocumentAsync({ type: ['application/pdf', 'text/plain'], copyToCacheDirectory: true });
      if (!result.canceled) {
        setUploadedFiles(prev => [...prev, result.assets[0]]);
      }
    } catch (error) { console.error("Picker error:", error); }
  };

  const uploadFileToBackend = async (file: DocumentPickerAsset, sessionId: string): Promise<string> => {
    const formData = new FormData();
    formData.append('session_id', sessionId);
    if (Platform.OS === 'web' && file.file) {
      formData.append('file', file.file, file.name);
    } else {
      formData.append('file', { uri: file.uri, name: file.name, type: file.mimeType || 'application/pdf' } as any);
    }
    const res = await fetch(`${API_BASE}/upload`, { method: 'POST', body: formData, headers: { 'Accept': 'application/json' } });
    if (!res.ok) throw new Error(file.name);
    return file.name;
  };

  // 2. SESSION CREATION & LOADING
  const createWorkspaceSession = async () => {
    if (!configName.trim()) return;
    setLoadingState('creating');
    setConfigError('');

    try {
      const dbs = configDbNy ? "NY_Case_Law,User_Workspace" : "User_Workspace";
      const res = await fetch(`${API_BASE}/sessions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: configName, description: configDesc, databases: dbs })
      });
      if (!res.ok) throw new Error('Could not create session. Check that the backend is running at localhost:8000.');
      const data = await res.json();
      const newSessionId = data.id;

      const successfulFiles: string[] = [];
      const failedFiles: string[] = [];
      for (const file of uploadedFiles) {
        try {
          await uploadFileToBackend(file, newSessionId);
          successfulFiles.push(file.name);
        } catch {
          failedFiles.push(file.name);
        }
      }

      const uploadNote = failedFiles.length > 0
        ? `\n\n⚠️ Failed to index: ${failedFiles.join(', ')}. These files may be corrupted or unsupported.`
        : successfulFiles.length > 0 ? ` ${successfulFiles.length} file(s) indexed.` : '';

      setActiveSessionId(newSessionId);
      setSessionName(data.name);
      setIndexedFiles(successfulFiles);
      setMessages([{ id: Date.now().toString(), role: 'assistant', text: `Workspace initialized for **${data.name}**.${uploadNote}` }]);
      setCases([]);
      setCurrentScreen('workspace');

      setConfigName('');
      setConfigDesc('');
      setUploadedFiles([]);
    } catch (error: any) {
      setConfigError(error.message || 'Something went wrong. Please try again.');
    } finally {
      setLoadingState('');
    }
  };

  const loadExistingSession = async (session) => {
    setActiveSessionId(session.id);
    setSessionName(session.name);
    setCurrentScreen('workspace');
    setCases([]);
    setIndexedFiles([]);

    try {
      const res = await fetch(`${API_BASE}/sessions/${session.id}/messages`);
      const data = await res.json();
      if (data.messages && data.messages.length > 0) {
        setMessages(data.messages);

        const lastAssistantMsg = [...data.messages].reverse().find(m => m.role === 'assistant' && m.graph_state && m.graph_state !== "[]");
        if (lastAssistantMsg) {
          setCases(JSON.parse(lastAssistantMsg.graph_state));
        }
      } else {
        setMessages([{ id: Date.now().toString(), role: 'assistant', text: `Loaded workspace for: **${session.name}**.` }]);
      }
    } catch (error) {
      setMessages([{ id: Date.now().toString(), role: 'assistant', text: `⚠️ Failed to load chat history. Check that the backend is running at localhost:8000.` }]);
    }
  };

  const saveMessageToDB = async (role, content, graphState = "[]") => {
    if (!activeSessionId) return;
    try {
      await fetch(`${API_BASE}/messages`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: activeSessionId, role, content, graph_state: graphState })
      });
    } catch (error) { console.error("Error saving message", error); }
  };

  // 3. WORKSPACE ANALYSIS & TIME TRAVEL
  const handleAnalyze = async () => {
    if (!argument.trim() && uploadedFiles.length === 0) return;

    let phase = 'uploading';

    // Upload files staged in the workspace input
    if (uploadedFiles.length > 0) {
      setLoadingState('searching');
      const newlyIndexed: string[] = [];
      for (const file of uploadedFiles) {
        try {
          await uploadFileToBackend(file, activeSessionId!);
          newlyIndexed.push(file.name);
        } catch {
          setMessages(prev => [...prev, {
            id: Date.now().toString(),
            role: 'assistant',
            text: `⚠️ Failed to index **${file.name}**. The file may be corrupted or unsupported.`
          }]);
        }
      }
      if (newlyIndexed.length > 0) {
        setIndexedFiles(prev => [...prev, ...newlyIndexed]);
      }
    }

    const userText = argument;
    const fileNames = uploadedFiles.map(f => f.name);
    const userMessage = { id: Date.now().toString(), role: 'user', text: userText, files: fileNames };

    setMessages(prev => [...prev, userMessage]);
    setArgument('');
    setUploadedFiles([]);

    await saveMessageToDB('user', userText);

    setLoadingState('searching');

    try {
      let contextText = "";
      let finalCasesState = [...cases];

      if (userText.trim()) {
        phase = 'searching';
        const searchResponse = await fetch(`${API_BASE}/search`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ session_id: activeSessionId, argument: userText }),
        });
        if (!searchResponse.ok) throw new Error('search');
        const searchData = await searchResponse.json();

        // CUMULATIVE GRAPH
        setCases(prevCases => {
          const updatedCases = [...prevCases];
          searchData.cases.forEach(incomingCase => {
            const existingIndex = updatedCases.findIndex(c => c.id === incomingCase.id);
            if (existingIndex >= 0) {
              updatedCases[existingIndex].hitCount = (updatedCases[existingIndex].hitCount || 1) + 1;
              updatedCases[existingIndex].distance = (updatedCases[existingIndex].distance + incomingCase.distance) / 2;
            } else {
              updatedCases.push({ ...incomingCase, hitCount: 1 });
            }
          });
          const sortedCases = updatedCases.sort((a, b) => {
            if ((b.hitCount || 1) !== (a.hitCount || 1)) return (b.hitCount || 1) - (a.hitCount || 1);
            return a.distance - b.distance;
          });
          finalCasesState = sortedCases;
          return sortedCases;
        });

        if (searchData.cases.length > 0 && !selectedCase) setSelectedCase(searchData.cases[0]);
        contextText = searchData.context_text;
      }

      phase = 'generating';
      setLoadingState('generating');
      const aiMsgId = (Date.now() + 1).toString();
      setMessages(prev => [...prev, { id: aiMsgId, role: 'assistant', text: '', graph_state: JSON.stringify(finalCasesState) }]);

      const genResponse = await fetch(`${API_BASE}/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: activeSessionId, argument: userText, context_text: contextText }),
      });
      if (!genResponse.ok) throw new Error('generate');

      const reader = genResponse.body!.getReader();
      const decoder = new TextDecoder();
      let done = false;
      let finalAiText = "";

      while (!done) {
        const { value, done: readerDone } = await reader.read();
        done = readerDone;
        if (value) {
          const chunk = decoder.decode(value, { stream: true });
          finalAiText += chunk;
          setMessages(prev => prev.map(msg =>
            msg.id === aiMsgId ? { ...msg, text: msg.text + chunk } : msg
          ));
        }
      }

      await saveMessageToDB('assistant', finalAiText, JSON.stringify(finalCasesState));

    } catch (error: any) {
      const msg = phase === 'searching'
        ? "⚠️ Case law search failed. Check that the backend is running at localhost:8000."
        : "⚠️ Response generation failed. Check that Ollama is running with the Llama3 model (`ollama run llama3`).";
      setMessages(prev => [...prev, { id: Date.now().toString(), role: 'assistant', text: msg }]);
    } finally {
      setLoadingState('');
    }
  };

  const handleZoomIn = () => setZoomScale(prev => Math.min(prev + 0.25, 2.5));
  const handleZoomOut = () => setZoomScale(prev => Math.max(prev - 0.25, 0.5));

  const getRelevance = (index) => {
    if (index === 0) return { label: 'High Match', color: '#059669', bg: '#D1FAE5' };
    if (index < 3) return { label: 'Strong Match', color: '#D97706', bg: '#FEF3C7' };
    return { label: 'Contextual', color: '#64748B', bg: '#F1F5F9' };
  };

  const renderGraphEdges = () => {
    const CENTER = 150;
    return (
      <Svg style={StyleSheet.absoluteFill}>
        {cases.map((c, index) => {
          const hitMultiplier = c.hitCount ? Math.max(1, 4 - c.hitCount) : 3;
          const radius = 40 + (index * 15 * hitMultiplier);
          const angle = index * ((2 * Math.PI) / cases.length);
          const x = CENTER + radius * Math.cos(angle);
          const y = CENTER + radius * Math.sin(angle);
          const isSelected = selectedCase?.id === c.id;
          const isCriticalNode = c.hitCount && c.hitCount > 1;
          return (
            <Line
              key={c.id}
              x1={CENTER} y1={CENTER} x2={x} y2={y}
              stroke={isSelected ? "#2563EB" : (isCriticalNode ? "#D97706" : "#E2E8F0")}
              strokeWidth={isSelected || isCriticalNode ? 2 : 1}
              strokeDasharray={isSelected || isCriticalNode ? "" : "4,4"}
            />
          );
        })}
      </Svg>
    );
  };

  const renderGraphNodes = () => {
    const CENTER = 150;
    return cases.map((c, index) => {
      const hitMultiplier = c.hitCount ? Math.max(1, 4 - c.hitCount) : 3;
      const radius = 40 + (index * 15 * hitMultiplier);
      const angle = index * ((2 * Math.PI) / cases.length);
      const x = CENTER + radius * Math.cos(angle);
      const y = CENTER + radius * Math.sin(angle);
      const isSelected = selectedCase?.id === c.id;
      const isCriticalNode = c.hitCount && c.hitCount > 1;
      const rel = getRelevance(index);
      return (
        <TouchableOpacity key={c.id} style={[styles.nodeWrapper, { left: x - 15, top: y - 15 }]} onPress={() => setSelectedCase(c)} activeOpacity={0.7}>
          <View style={[
            styles.graphNode,
            { borderColor: isSelected ? '#2563EB' : (isCriticalNode ? '#D97706' : rel.color), backgroundColor: rel.bg, transform: [{ scale: isCriticalNode ? 1.3 : 1 }] }
          ]}>
            <View style={[styles.nodeDot, { backgroundColor: isSelected ? '#2563EB' : (isCriticalNode ? '#D97706' : rel.color) }]} />
          </View>
          <View style={styles.nodeLabelContainer}>
            <Text style={[styles.nodeLabelText, isSelected && styles.nodeLabelTextSelected]} numberOfLines={1}>
              {isCriticalNode && "🔥 "} {c.id}
            </Text>
            <Text style={styles.nodeDistanceText}>{(c.hitCount ?? 0) > 1 ? `Connections: ${c.hitCount}` : `Dist: ${c.distance.toFixed(2)}`}</Text>
          </View>
        </TouchableOpacity>
      );
    });
  };

  // ==========================================
  // PAGE 1: THE LOBBY
  // ==========================================
  if (currentScreen === 'lobby') {
    return (
      <View style={styles.lobbyWrapper}>
        <View style={styles.lobbySidebar}>
          <Text style={styles.sidebarSection}>Recent Matters</Text>
          <ScrollView>
            {savedSessions.length === 0 ? (
              <Text style={{ color: '#94A3B8', fontSize: 13, marginTop: 10 }}>No sessions found.</Text>
            ) : (
              savedSessions.map(s => (
                <TouchableOpacity key={s.id} style={styles.historyItem} onPress={() => loadExistingSession(s)}>
                  <Ionicons name="folder-outline" size={16} color="#94A3B8" />
                  <View style={{ marginLeft: 10, flex: 1 }}>
                    <Text style={styles.historyText} numberOfLines={1}>{s.name}</Text>
                    <Text style={{ color: '#64748B', fontSize: 11 }}>{new Date(s.created_at).toLocaleDateString()}</Text>
                  </View>
                </TouchableOpacity>
              ))
            )}
          </ScrollView>
        </View>

        <View style={styles.lobbyMain}>
          <View style={styles.lobbyHeader}>
            <View style={styles.lobbyLogo}><Ionicons name="scale" size={32} color="#FFFFFF" /></View>
            <Text style={styles.lobbyTitle}>Legal Scribe Workspace</Text>
          </View>
          <TouchableOpacity style={styles.primaryActionCard} onPress={() => setCurrentScreen('config')} activeOpacity={0.8}>
            <View style={styles.actionIconContainer}><Ionicons name="add" size={28} color="#2563EB" /></View>
            <Text style={styles.actionTitle}>Create New Matter</Text>
            <Text style={styles.actionDesc}>Initialize a new workspace, define databases, and upload case files.</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  }

  // ==========================================
  // PAGE 2: SESSION CONFIGURATION FORM
  // ==========================================
  if (currentScreen === 'config') {
    return (
      <View style={[styles.lobbyWrapper, { alignItems: 'center', justifyContent: 'center' }]}>
        <View style={styles.configContainer}>
          <TouchableOpacity style={styles.backBtn} onPress={() => setCurrentScreen('lobby')}>
            <Ionicons name="arrow-back" size={20} color="#0F172A" />
            <Text style={styles.backBtnText}>Back to Lobby</Text>
          </TouchableOpacity>

          <Text style={styles.configTitle}>Configure Workspace</Text>

          <View style={styles.formGroup}>
            <Text style={styles.label}>Matter Name</Text>
            <TextInput style={styles.textInput} placeholder="e.g., Smith v. Jones - Summary Judgment" value={configName} onChangeText={setConfigName} />
          </View>

          <View style={styles.formGroup}>
            <Text style={styles.label}>Description (Optional)</Text>
            <TextInput style={[styles.textInput, { height: 80 }]} multiline placeholder="Brief overview of the legal issues..." value={configDesc} onChangeText={setConfigDesc} />
          </View>

          <View style={styles.formGroup}>
            <Text style={styles.label}>General Law Databases</Text>
            <TouchableOpacity style={[styles.dbToggle, configDbNy && styles.dbToggleActive]} onPress={() => setConfigDbNy(!configDbNy)}>
              <Ionicons name={configDbNy ? "checkmark-circle" : "ellipse-outline"} size={20} color={configDbNy ? "#2563EB" : "#94A3B8"} />
              <Text style={styles.dbToggleText}>NY Court of Appeals Precedent</Text>
            </TouchableOpacity>
          </View>

          <View style={styles.formGroup}>
            <Text style={styles.label}>Initial Case Files</Text>
            <TouchableOpacity style={styles.uploadBox} onPress={pickFile}>
              <Ionicons name="document-attach" size={24} color="#64748B" />
              <Text style={styles.uploadBoxText}>Browse files from Mac</Text>
            </TouchableOpacity>
            {uploadedFiles.map((file, i) => (
              <View key={i} style={styles.fileListRow}>
                <Text style={styles.fileListText} numberOfLines={1}>📎 {file.name}</Text>
                <TouchableOpacity onPress={() => removeFile(i)} hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}>
                  <Ionicons name="close-circle" size={18} color="#94A3B8" />
                </TouchableOpacity>
              </View>
            ))}
          </View>

          <TouchableOpacity style={[styles.createBtn, (!configName.trim() || loadingState !== '') && styles.createBtnDisabled]} onPress={createWorkspaceSession} disabled={!configName.trim() || loadingState !== ''}>
            {loadingState === 'creating' ? <ActivityIndicator color="#FFF" /> : <Text style={styles.createBtnText}>Launch Workspace</Text>}
          </TouchableOpacity>

          {configError !== '' && (
            <Text style={styles.configErrorText}>{configError}</Text>
          )}
        </View>
      </View>
    );
  }

  // ==========================================
  // PAGE 3: THE WORKSPACE
  // ==========================================
  return (
    <View style={styles.appWrapper}>

      {/* SIDEBAR NAVIGATION */}
      <View style={styles.sidebar}>
        <View style={styles.brandContainer}>
          <Ionicons name="scale-outline" size={24} color="#FFFFFF" />
          <Text style={styles.brandText}>Legal Scribe</Text>
        </View>

        <TouchableOpacity style={styles.newChatBtn} onPress={() => setCurrentScreen('lobby')}>
          <Ionicons name="home-outline" size={18} color="#0F172A" />
          <Text style={styles.newChatText}>Return to Lobby</Text>
        </TouchableOpacity>

        <Text style={styles.sidebarSection}>Current Matter</Text>
        <View style={styles.activeSessionItem}>
          <Ionicons name="document-text" size={16} color="#38BDF8" />
          <Text style={styles.activeSessionText} numberOfLines={2}>{sessionName}</Text>
        </View>

        {/* INDEXED FILES LIST */}
        <Text style={[styles.sidebarSection, { marginTop: 24 }]}>Indexed Files</Text>
        {indexedFiles.length === 0 ? (
          <Text style={styles.indexedFilesEmpty}>No files uploaded this session.</Text>
        ) : (
          indexedFiles.map((name, i) => (
            <View key={i} style={styles.indexedFileItem}>
              <Ionicons name="document-text-outline" size={13} color="#38BDF8" />
              <Text style={styles.indexedFileName} numberOfLines={1}>{name}</Text>
            </View>
          ))
        )}
      </View>

      <KeyboardAvoidingView behavior={Platform.OS === 'ios' ? 'padding' : 'height'} style={styles.container}>
        {/* PANE 1: AI Chat Session */}
        <View style={styles.pane}>
          <View style={styles.paneHeader}>
            <Ionicons name="chatbubbles-outline" size={20} color="#0F172A" />
            <Text style={styles.header}>Session Workspace</Text>
          </View>

          <ScrollView style={styles.chatContainer} ref={scrollViewRef} onContentSizeChange={() => scrollViewRef.current?.scrollToEnd({ animated: true })}>
            {messages.map((msg) => (
              <View key={msg.id} style={[styles.chatBubbleWrapper, msg.role === 'user' ? styles.userBubbleWrapper : styles.aiBubbleWrapper]}>
                {msg.role === 'assistant' && (
                  <View style={styles.aiAvatar}>
                    <Ionicons name="scale" size={14} color="#FFFFFF" />
                  </View>
                )}
                <View style={[styles.chatBubble, msg.role === 'user' ? styles.userBubble : styles.aiBubble]}>
                  {msg.files && msg.files.length > 0 && (
                    <View style={styles.bubbleFilesContainer}>
                      {msg.files.map((file, i) => (
                        <View key={i} style={styles.bubbleFileTag}>
                          <Ionicons name="document-attach" size={12} color="#1E3A8A" />
                          <Text style={styles.bubbleFileText}>{file}</Text>
                        </View>
                      ))}
                    </View>
                  )}
                  {msg.role === 'user' ? (
                    <Text style={styles.userBubbleText}>{msg.text}</Text>
                  ) : (
                    <>
                      <Markdown style={markdownStyles}>
                        {msg.text + (loadingState === 'generating' && msg.text.length > 0 && msg.id === messages[messages.length - 1]?.id ? ' ▋' : '')}
                      </Markdown>

                      {/* --- GRAPH CHECKPOINT TIME TRAVEL BUTTON --- */}
                      {msg.graph_state && msg.graph_state !== "[]" && (
                        <TouchableOpacity
                          style={styles.checkpointBtn}
                          onPress={() => setCases(JSON.parse(msg.graph_state!))}
                        >
                          <Ionicons name="git-pull-request-outline" size={14} color="#2563EB" />
                          <Text style={styles.checkpointText}>View Graph Checkpoint</Text>
                        </TouchableOpacity>
                      )}
                    </>
                  )}
                </View>
              </View>
            ))}
            {loadingState === 'searching' && (
              <View style={styles.statusRow}>
                <ActivityIndicator size="small" color="#2563EB" />
                <Text style={styles.statusText}>Searching General Law & Uploads...</Text>
              </View>
            )}
          </ScrollView>

          <View style={styles.inputWrapper}>
            {uploadedFiles.length > 0 && (
              <View style={styles.fileTagsContainer}>
                {uploadedFiles.map((file, i) => (
                  <View key={i} style={styles.fileTag}>
                    <Ionicons name="document-attach" size={12} color="#475569" />
                    <Text style={styles.fileTagText} numberOfLines={1}>{file.name}</Text>
                    <TouchableOpacity onPress={() => removeFile(i)} style={styles.fileTagRemove} hitSlop={{ top: 6, bottom: 6, left: 6, right: 6 }}>
                      <Ionicons name="close" size={12} color="#94A3B8" />
                    </TouchableOpacity>
                  </View>
                ))}
              </View>
            )}
            <View style={styles.inputRow}>
              <TouchableOpacity style={styles.iconButton} onPress={pickFile}>
                <Ionicons name="add-circle" size={26} color="#64748B" />
              </TouchableOpacity>
              <TextInput style={styles.input} multiline placeholder="Ask a question or provide direction..." placeholderTextColor="#94A3B8" value={argument} onChangeText={setArgument} />
              <TouchableOpacity style={[styles.sendButton, (!argument.trim() && uploadedFiles.length === 0) && styles.sendButtonDisabled]} onPress={handleAnalyze} disabled={(!argument.trim() && uploadedFiles.length === 0) || loadingState !== ''}>
                <Ionicons name="arrow-up" size={18} color="#FFFFFF" />
              </TouchableOpacity>
            </View>
          </View>
        </View>

        {/* PANE 2: Graph */}
        <View style={[styles.pane, styles.middlePane]}>
          <View style={styles.paneHeader}>
            <Ionicons name="git-network-outline" size={20} color="#0F172A" />
            <Text style={styles.header}>Semantic Network</Text>
          </View>
          <View style={styles.graphContainer}>
            <View style={[styles.zoomArea, { transform: [{ scale: zoomScale }] }]}>
              {cases.length > 0 ? (
                <>
                  {renderGraphEdges()}
                  {renderGraphNodes()}
                  <View style={[styles.centerNodeWrapper, { left: 150 - 25, top: 150 - 25 }]}>
                    <View style={styles.centerNode}>
                      <Text style={styles.centerNodeText}>Current</Text>
                    </View>
                  </View>
                </>
              ) : (
                <View style={styles.emptyState}>
                  <Ionicons name="compass-outline" size={40} color="#E2E8F0" />
                  <Text style={styles.placeholderText}>No network generated yet.</Text>
                </View>
              )}
            </View>

            {/* TRASH CAN AND ZOOM CONTROLS */}
            <View style={styles.zoomControls}>
              <TouchableOpacity style={styles.zoomButton} onPress={() => setCases([])}>
                <Ionicons name="trash-outline" size={18} color="#64748B" />
              </TouchableOpacity>
              <View style={styles.zoomDivider} />
              <TouchableOpacity style={styles.zoomButton} onPress={handleZoomIn}><Text style={styles.zoomButtonText}>+</Text></TouchableOpacity>
              <View style={styles.zoomDivider} />
              <TouchableOpacity style={styles.zoomButton} onPress={handleZoomOut}><Text style={styles.zoomButtonText}>-</Text></TouchableOpacity>
            </View>
          </View>

          <ScrollView style={styles.caseList} showsVerticalScrollIndicator={false}>
            {cases.map((c, index) => {
              const rel = getRelevance(index);
              const isSelected = selectedCase?.id === c.id;
              return (
                <TouchableOpacity key={c.id} style={[styles.caseCard, isSelected && styles.selectedCard]} onPress={() => setSelectedCase(c)} activeOpacity={0.7}>
                  <View style={styles.cardHeader}>
                    <Text style={styles.caseTitle} numberOfLines={1}>{c.id}</Text>
                    <View style={[styles.badge, { backgroundColor: isSelected ? '#2563EB' : rel.bg }]}>
                      <Text style={[styles.badgeText, { color: isSelected ? '#FFF' : rel.color }]}>{rel.label}</Text>
                    </View>
                  </View>
                  <Text style={styles.caseDate}>{c.date}</Text>
                </TouchableOpacity>
              );
            })}
          </ScrollView>
        </View>

        {/* PANE 3: Doc Viewer */}
        <View style={styles.pane}>
          <View style={styles.paneHeader}>
            <Ionicons name="book-outline" size={20} color="#0F172A" />
            <Text style={styles.header}>Primary Source</Text>
          </View>
          <ScrollView style={styles.documentViewer} showsVerticalScrollIndicator={false}>
            {selectedCase ? (
              <>
                <Text style={styles.docTitle}>{selectedCase.id}</Text>
                <Text style={styles.docMeta}>Decided: {selectedCase.date}</Text>
                <View style={styles.divider} />
                <Markdown style={markdownStyles}>{selectedCase.text}</Markdown>
              </>
            ) : (
              <View style={styles.emptyState}>
                <Ionicons name="search-outline" size={40} color="#E2E8F0" />
                <Text style={styles.placeholderText}>Select a node to review the court opinion.</Text>
              </View>
            )}
          </ScrollView>
        </View>
      </KeyboardAvoidingView>
    </View>
  );
}

// --- STYLESHEETS ---
const styles = StyleSheet.create({
  // Lobby Styles
  lobbyWrapper: { flex: 1, flexDirection: 'row', backgroundColor: '#F8FAFC' },
  lobbyMain: { flex: 1, alignItems: 'center', justifyContent: 'center' },
  lobbySidebar: { width: 260, backgroundColor: '#0F172A', padding: 20, paddingTop: 40 },
  lobbyHeader: { alignItems: 'center', marginBottom: 50 },
  lobbyLogo: { width: 64, height: 64, borderRadius: 16, backgroundColor: '#0F172A', alignItems: 'center', justifyContent: 'center', marginBottom: 20 },
  lobbyTitle: { fontSize: 36, fontWeight: '800', color: '#0F172A', letterSpacing: -1, marginBottom: 10 },
  primaryActionCard: { width: 350, backgroundColor: '#FFFFFF', padding: 30, borderRadius: 20, borderWidth: 2, borderColor: '#BFDBFE', shadowColor: '#2563EB', shadowOpacity: 0.1, shadowRadius: 20, shadowOffset: { width: 0, height: 10 }, alignItems: 'center' },
  actionIconContainer: { width: 56, height: 56, borderRadius: 28, backgroundColor: '#EFF6FF', alignItems: 'center', justifyContent: 'center', marginBottom: 20 },
  actionTitle: { fontSize: 20, fontWeight: '700', color: '#0F172A', marginBottom: 8 },
  actionDesc: { fontSize: 14, color: '#64748B', lineHeight: 22, textAlign: 'center' },
  historyItem: { flexDirection: 'row', alignItems: 'center', paddingVertical: 12, borderBottomWidth: 1, borderBottomColor: '#1E293B' },
  historyText: { color: '#CBD5E1', fontSize: 14, fontWeight: '500' },

  // Config Styles
  configContainer: { width: 500, backgroundColor: '#FFFFFF', padding: 40, borderRadius: 16, shadowColor: '#000', shadowOpacity: 0.05, shadowRadius: 15 },
  backBtn: { flexDirection: 'row', alignItems: 'center', marginBottom: 30 },
  backBtnText: { marginLeft: 8, fontSize: 15, fontWeight: '600', color: '#0F172A' },
  configTitle: { fontSize: 28, fontWeight: '700', color: '#0F172A', marginBottom: 30 },
  formGroup: { marginBottom: 24 },
  label: { fontSize: 14, fontWeight: '600', color: '#475569', marginBottom: 8 },
  textInput: { borderWidth: 1, borderColor: '#E2E8F0', borderRadius: 8, padding: 12, fontSize: 15, backgroundColor: '#F8FAFC' },
  dbToggle: { flexDirection: 'row', alignItems: 'center', padding: 16, borderWidth: 1, borderColor: '#E2E8F0', borderRadius: 8, backgroundColor: '#F8FAFC' },
  dbToggleActive: { borderColor: '#BFDBFE', backgroundColor: '#EFF6FF' },
  dbToggleText: { marginLeft: 12, fontSize: 15, fontWeight: '500', color: '#0F172A' },
  uploadBox: { borderWidth: 1, borderStyle: 'dashed', borderColor: '#CBD5E1', borderRadius: 8, padding: 20, alignItems: 'center', backgroundColor: '#F8FAFC' },
  uploadBoxText: { marginTop: 8, color: '#64748B', fontWeight: '500' },
  fileListRow: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginTop: 10, paddingHorizontal: 4 },
  fileListText: { color: '#2563EB', fontSize: 13, fontWeight: '500', flex: 1, marginRight: 8 },
  createBtn: { backgroundColor: '#0F172A', padding: 16, borderRadius: 8, alignItems: 'center', marginTop: 10 },
  createBtnDisabled: { backgroundColor: '#94A3B8' },
  createBtnText: { color: '#FFFFFF', fontWeight: '600', fontSize: 16 },
  configErrorText: { color: '#EF4444', fontSize: 13, fontWeight: '500', marginTop: 12, textAlign: 'center' },

  // Workspace Styles
  appWrapper: { flex: 1, flexDirection: 'row', backgroundColor: '#F8FAFC' },
  sidebar: { width: 260, backgroundColor: '#0F172A', padding: 20, paddingTop: 40 },
  brandContainer: { flexDirection: 'row', alignItems: 'center', marginBottom: 40 },
  brandText: { color: '#FFFFFF', fontSize: 20, fontWeight: '700', marginLeft: 10, letterSpacing: -0.5 },
  newChatBtn: { backgroundColor: '#F8FAFC', flexDirection: 'row', alignItems: 'center', justifyContent: 'center', padding: 12, borderRadius: 8, marginBottom: 30 },
  newChatText: { color: '#0F172A', fontWeight: '600', marginLeft: 8 },
  sidebarSection: { color: '#64748B', fontSize: 12, fontWeight: '600', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 12 },
  activeSessionItem: { flexDirection: 'row', alignItems: 'center', backgroundColor: '#1E293B', padding: 12, borderRadius: 8 },
  activeSessionText: { color: '#FFFFFF', fontSize: 14, fontWeight: '600', marginLeft: 10, flex: 1 },
  indexedFileItem: { flexDirection: 'row', alignItems: 'center', paddingVertical: 7, borderBottomWidth: 1, borderBottomColor: '#1E293B' },
  indexedFileName: { color: '#94A3B8', fontSize: 12, marginLeft: 8, flex: 1 },
  indexedFilesEmpty: { color: '#475569', fontSize: 12, fontStyle: 'italic' },

  container: { flex: 1, flexDirection: 'row', padding: 16 },
  pane: { flex: 1, backgroundColor: '#FFFFFF', margin: 8, borderRadius: 16, padding: 20, shadowColor: '#000', shadowOpacity: 0.03, shadowRadius: 10, shadowOffset: { width: 0, height: 2 }, borderWidth: 1, borderColor: '#E2E8F0', display: 'flex', flexDirection: 'column' },
  middlePane: { flex: 1.2 },
  paneHeader: { flexDirection: 'row', alignItems: 'center', marginBottom: 16 },
  header: { fontSize: 18, fontWeight: '700', color: '#0F172A', marginLeft: 10 },

  chatContainer: { flex: 1, marginBottom: 16 },
  chatBubbleWrapper: { marginBottom: 20, flexDirection: 'row', alignItems: 'flex-start' },
  userBubbleWrapper: { justifyContent: 'flex-end' },
  aiBubbleWrapper: { justifyContent: 'flex-start' },
  aiAvatar: { width: 28, height: 28, borderRadius: 14, backgroundColor: '#0F172A', alignItems: 'center', justifyContent: 'center', marginRight: 12, marginTop: 4 },
  chatBubble: { maxWidth: '85%', padding: 16, borderRadius: 16 },
  userBubble: { backgroundColor: '#EFF6FF', borderBottomRightRadius: 4, borderWidth: 1, borderColor: '#BFDBFE' },
  aiBubble: { backgroundColor: '#FFFFFF', borderBottomLeftRadius: 4, borderWidth: 1, borderColor: '#E2E8F0' },
  userBubbleText: { fontSize: 15, color: '#1E3A8A', lineHeight: 24 },
  bubbleFilesContainer: { flexDirection: 'row', flexWrap: 'wrap', marginBottom: 8 },
  bubbleFileTag: { flexDirection: 'row', alignItems: 'center', backgroundColor: '#DBEAFE', paddingHorizontal: 8, paddingVertical: 4, borderRadius: 6, marginRight: 6, marginBottom: 4 },
  bubbleFileText: { fontSize: 12, color: '#1E3A8A', marginLeft: 4, fontWeight: '600' },

  checkpointBtn: { flexDirection: 'row', alignItems: 'center', marginTop: 12, paddingVertical: 6, paddingHorizontal: 12, backgroundColor: '#EFF6FF', borderRadius: 8, alignSelf: 'flex-start', borderWidth: 1, borderColor: '#BFDBFE' },
  checkpointText: { marginLeft: 6, fontSize: 12, fontWeight: '600', color: '#2563EB' },

  emptyState: { flex: 1, alignItems: 'center', justifyContent: 'center', padding: 20, marginTop: 60 },
  placeholderText: { color: '#64748B', fontSize: 15, textAlign: 'center', lineHeight: 24, marginBottom: 24 },
  statusRow: { flexDirection: 'row', alignItems: 'center', marginVertical: 10, padding: 12, backgroundColor: '#F8FAFC', borderRadius: 8, alignSelf: 'flex-start' },
  statusText: { marginLeft: 12, fontSize: 14, color: '#475569', fontWeight: '500' },

  inputWrapper: { backgroundColor: '#FFFFFF', borderWidth: 1, borderColor: '#E2E8F0', borderRadius: 16, padding: 12 },
  fileTagsContainer: { flexDirection: 'row', flexWrap: 'wrap', marginBottom: 10, paddingHorizontal: 4 },
  fileTag: { flexDirection: 'row', alignItems: 'center', backgroundColor: '#F1F5F9', paddingHorizontal: 10, paddingVertical: 6, borderRadius: 8, marginRight: 8, marginBottom: 6, maxWidth: 200 },
  fileTagText: { fontSize: 13, color: '#334155', marginLeft: 6, fontWeight: '500', flex: 1 },
  fileTagRemove: { marginLeft: 6 },
  inputRow: { flexDirection: 'row', alignItems: 'flex-end' },
  iconButton: { padding: 8, paddingBottom: 10 },
  input: { flex: 1, minHeight: 44, maxHeight: 120, fontSize: 15, color: '#0F172A', paddingTop: 12, paddingBottom: 12, paddingHorizontal: 12 },
  sendButton: { backgroundColor: '#2563EB', width: 36, height: 36, borderRadius: 18, alignItems: 'center', justifyContent: 'center', marginBottom: 6, marginRight: 4 },
  sendButtonDisabled: { backgroundColor: '#E2E8F0' },

  graphContainer: { height: 320, backgroundColor: '#F8FAFC', borderRadius: 12, marginBottom: 20, alignItems: 'center', justifyContent: 'center', position: 'relative', overflow: 'hidden', borderWidth: 1, borderColor: '#E2E8F0' },
  zoomArea: { width: 300, height: 300, alignItems: 'center', justifyContent: 'center' },
  nodeWrapper: { position: 'absolute', width: 30, height: 30, alignItems: 'center', justifyContent: 'center', zIndex: 10, overflow: 'visible' },
  graphNode: { width: 20, height: 20, borderRadius: 10, borderWidth: 2, alignItems: 'center', justifyContent: 'center', backgroundColor: '#FFFFFF' },
  nodeDot: { width: 8, height: 8, borderRadius: 4 },
  nodeLabelContainer: { position: 'absolute', top: 24, width: 100, alignItems: 'center', backgroundColor: 'rgba(255, 255, 255, 0.9)', paddingVertical: 2, paddingHorizontal: 4, borderRadius: 4 },
  nodeLabelText: { fontSize: 11, fontWeight: '600', color: '#1E293B', textAlign: 'center' },
  nodeLabelTextSelected: { color: '#2563EB', fontWeight: '700' },
  nodeDistanceText: { fontSize: 9, color: '#64748B', textAlign: 'center', marginTop: 1 },
  centerNodeWrapper: { position: 'absolute', width: 50, height: 50, alignItems: 'center', justifyContent: 'center', zIndex: 5 },
  centerNode: { width: 44, height: 44, borderRadius: 22, backgroundColor: '#0F172A', alignItems: 'center', justifyContent: 'center' },
  centerNodeText: { color: '#FFFFFF', fontSize: 10, fontWeight: '700' },
  zoomControls: { position: 'absolute', bottom: 12, right: 12, backgroundColor: '#FFFFFF', borderRadius: 8, flexDirection: 'row', alignItems: 'center', borderWidth: 1, borderColor: '#E2E8F0' },
  zoomButton: { paddingHorizontal: 12, paddingVertical: 6 },
  zoomButtonText: { fontSize: 16, fontWeight: '500', color: '#0F172A' },
  zoomDivider: { width: 1, height: 16, backgroundColor: '#E2E8F0' },

  caseList: { flex: 1 },
  caseCard: { backgroundColor: '#FFFFFF', padding: 16, borderRadius: 12, marginBottom: 10, borderWidth: 1, borderColor: '#E2E8F0' },
  selectedCard: { borderColor: '#2563EB', backgroundColor: '#EFF6FF' },
  cardHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 },
  caseTitle: { fontWeight: '600', fontSize: 14, color: '#0F172A', flex: 1, marginRight: 10 },
  badge: { paddingHorizontal: 8, paddingVertical: 4, borderRadius: 6 },
  badgeText: { fontSize: 10, fontWeight: '700' },
  caseDate: { fontSize: 12, color: '#64748B' },

  documentViewer: { flex: 1 },
  docTitle: { fontSize: 18, fontWeight: '700', color: '#0F172A', marginBottom: 6, letterSpacing: -0.3 },
  docMeta: { fontSize: 13, color: '#64748B', marginBottom: 16 },
  divider: { height: 1, backgroundColor: '#E2E8F0', marginBottom: 20 },
});

const markdownStyles = StyleSheet.create({
  body: { fontSize: 15, lineHeight: 26, color: '#334155' },
  heading1: { fontSize: 18, fontWeight: '700', color: '#0F172A', marginTop: 16, marginBottom: 8 },
  heading2: { fontSize: 16, fontWeight: '600', color: '#0F172A', marginTop: 14, marginBottom: 6 },
  strong: { fontWeight: '700', color: '#0F172A' },
  blockquote: { borderLeftWidth: 3, borderLeftColor: '#2563EB', paddingLeft: 12, marginVertical: 8, backgroundColor: '#F8FAFC', paddingVertical: 8 },
});
