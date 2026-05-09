import React, { useState, useEffect, useRef } from 'react';
import { StyleSheet, Text, View, TextInput, TouchableOpacity, ScrollView, ActivityIndicator, KeyboardAvoidingView, Platform } from 'react-native';
import type { TextStyle } from 'react-native';
import Svg, { Line } from 'react-native-svg';
import Markdown from 'react-native-markdown-display';
import { Ionicons } from '@expo/vector-icons';
import * as DocumentPicker from 'expo-document-picker';

const API_BASE = 'http://localhost:8000';

type Screen = 'lobby' | 'config' | 'library_manager' | 'workspace';
type LoadingState = '' | 'creating' | 'searching' | 'generating';

type Session = {
  id: string;
  name: string;
  description?: string;
  databases?: string;
  created_at: string;
};

type CaseItem = {
  id: string;
  date?: string;
  text: string;
  distance: number;
  hitCount?: number;
};

type Message = {
  id: string;
  role: 'user' | 'assistant';
  text: string;
  files?: string[];
  graph_state?: string;
};

type PickerAsset = DocumentPicker.DocumentPickerAsset & { file?: File };

export default function Index() {
  const [currentScreen, setCurrentScreen] = useState<Screen>('lobby');

  const [savedSessions, setSavedSessions] = useState<Session[]>([]);
  const [availableDatabases, setAvailableDatabases] = useState<string[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [sessionName, setSessionName] = useState('');

  const [configName, setConfigName] = useState('');
  const [configDesc, setConfigDesc] = useState('');
  const [selectedDatabases, setSelectedDatabases] = useState<string[]>([]);
  const [configError, setConfigError] = useState('');
  const [uploadedFiles, setUploadedFiles] = useState<PickerAsset[]>([]);
  const [indexedFiles, setIndexedFiles] = useState<string[]>([]);

  const [libraryName, setLibraryName] = useState('');
  const [libraryFiles, setLibraryFiles] = useState<PickerAsset[]>([]);
  const [isCreatingLibrary, setIsCreatingLibrary] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);

  const [messages, setMessages] = useState<Message[]>([]);
  const [argument, setArgument] = useState('');
  const [loadingState, setLoadingState] = useState<LoadingState>('');
  const [cases, setCases] = useState<CaseItem[]>([]);
  const [selectedCase, setSelectedCase] = useState<CaseItem | null>(null);
  const [zoomScale, setZoomScale] = useState(1);
  const scrollViewRef = useRef<ScrollView | null>(null);

  useEffect(() => {
    if (currentScreen === 'lobby' || currentScreen === 'config' || currentScreen === 'library_manager') {
      fetchSessions();
      fetchDatabases();
    }
  }, [currentScreen]);

  const fetchSessions = async () => {
    try {
      const res = await fetch(`${API_BASE}/sessions`);
      const data = await res.json();
      setSavedSessions(data.sessions);
    } catch (error) { console.error("Session load error", error); }
  };

  const fetchDatabases = async () => {
    try {
      const res = await fetch(`${API_BASE}/databases`);
      const data = await res.json();
      setAvailableDatabases(data.databases);
      if (data.databases.length > 0 && selectedDatabases.length === 0) {
        setSelectedDatabases([data.databases[0]]);
      }
    } catch (error) { console.error("Database load error", error); }
  };

  const removeFile = (index: number, setter: React.Dispatch<React.SetStateAction<PickerAsset[]>>) => {
    setter(prev => prev.filter((_, i) => i !== index));
  };

  const pickFile = async (
    setFilesState: React.Dispatch<React.SetStateAction<PickerAsset[]>>,
    multiple = false
  ) => {
    try {
      const result = await DocumentPicker.getDocumentAsync({
        type: ['application/pdf', 'text/plain', 'application/json', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'],
        copyToCacheDirectory: true,
        multiple,
      });
      if (!result.canceled) {
        setFilesState(prev => [...prev, ...result.assets]);
      }
    } catch (error) { console.error("Picker error", error); }
  };

  const uploadFileToBackend = async (file: PickerAsset, sessionId: string): Promise<string> => {
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

  const createFirmLibrary = async () => {
    if (!libraryName.trim() || libraryFiles.length === 0) return;
    setIsCreatingLibrary(true);
    setUploadProgress(0);
    try {
      const formData = new FormData();
      formData.append('db_name', libraryName);
      for (const file of libraryFiles) {
        const fileToUpload = Platform.OS === 'web' && file.file
          ? file.file
          : { uri: file.uri, name: file.name, type: file.mimeType || 'application/pdf' };
        formData.append('files', fileToUpload as any);
      }
      const res = await fetch(`${API_BASE}/databases/create`, { method: 'POST', body: formData });
      if (!res.ok) {
        const errorData = await res.json();
        console.error("FastAPI Error:", errorData);
        setIsCreatingLibrary(false);
        return;
      }
      if (!res.body) throw new Error('Missing response stream while creating library');
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let done = false;
      while (!done) {
        const { value, done: readerDone } = await reader.read();
        done = readerDone;
        if (value) {
          const chunk = decoder.decode(value);
          const lines = chunk.split('\n').filter(line => line.trim() !== '');
          for (const line of lines) {
            const data = JSON.parse(line);
            setUploadProgress(data.progress);
            if (data.status === "complete") {
              fetchDatabases();
              setTimeout(() => {
                setLibraryName('');
                setLibraryFiles([]);
                setUploadProgress(0);
                setCurrentScreen('lobby');
              }, 800);
            }
          }
        }
      }
    } catch (error) {
      console.error("Library creation error", error);
    } finally {
      setIsCreatingLibrary(false);
    }
  };

  const toggleDatabaseSelection = (dbName: string) => {
    setSelectedDatabases(prev =>
      prev.includes(dbName) ? prev.filter(db => db !== dbName) : [...prev, dbName]
    );
  };

  const createProject = async () => {
    if (!configName.trim()) return;
    setLoadingState('creating');
    setConfigError('');
    try {
      const dbsToUse = [...selectedDatabases, "user_workspace"].join(",");
      const res = await fetch(`${API_BASE}/sessions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: configName, description: configDesc, databases: dbsToUse })
      });
      if (!res.ok) throw new Error('Could not create session. Check that the backend is running at localhost:8000.');
      const data = await res.json();

      const successfulFiles: string[] = [];
      const failedFiles: string[] = [];
      for (const file of uploadedFiles) {
        try {
          await uploadFileToBackend(file, data.id);
          successfulFiles.push(file.name);
        } catch {
          failedFiles.push(file.name);
        }
      }

      const uploadNote = failedFiles.length > 0
        ? `\n\n⚠️ Failed to index: ${failedFiles.join(', ')}.`
        : successfulFiles.length > 0 ? ` ${successfulFiles.length} file(s) indexed.` : '';

      setActiveSessionId(data.id);
      setSessionName(data.name);
      setIndexedFiles(successfulFiles);
      setMessages([{ id: Date.now().toString(), role: 'assistant', text: `Workspace for **${data.name}** is ready.${uploadNote}` }]);
      setCases([]);
      setCurrentScreen('workspace');
      setUploadedFiles([]);
      setConfigName('');
      setConfigDesc('');
    } catch (error: any) {
      setConfigError(error.message || 'Something went wrong. Please try again.');
    } finally { setLoadingState(''); }
  };

  const openProject = async (session: Session) => {
    setActiveSessionId(session.id);
    setSessionName(session.name);
    setCurrentScreen('workspace');
    setCases([]);
    setIndexedFiles([]);
    try {
      const res = await fetch(`${API_BASE}/sessions/${session.id}/messages`);
      const data = await res.json();
      if (data.messages?.length > 0) {
        setMessages(data.messages);
        const lastWithGraph = [...data.messages].reverse().find((m: Message) => m.graph_state && m.graph_state !== "[]");
        if (lastWithGraph) setCases(JSON.parse(lastWithGraph.graph_state));
      } else {
        setMessages([{ id: Date.now().toString(), role: 'assistant', text: `Loaded workspace for: **${session.name}**.` }]);
      }
    } catch (error) {
      setMessages([{ id: Date.now().toString(), role: 'assistant', text: `⚠️ Failed to load chat history. Check that the backend is running at localhost:8000.` }]);
    }
  };

  const saveMessageToDB = async (role: string, content: string, graphState = "[]") => {
    if (!activeSessionId) return;
    try {
      await fetch(`${API_BASE}/messages`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: activeSessionId, role, content, graph_state: graphState })
      });
    } catch (error) { console.error("Error saving message", error); }
  };

  const handleAnalyze = async () => {
    if (!argument.trim() && uploadedFiles.length === 0) return;

    let phase = 'uploading';

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
      if (newlyIndexed.length > 0) setIndexedFiles(prev => [...prev, ...newlyIndexed]);
    }

    const userText = argument;
    const fileNames = uploadedFiles.map(f => f.name);
    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      text: userText,
      files: fileNames.length > 0 ? fileNames : undefined,
    };

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

        const updatedCases = [...cases];
        searchData.cases.forEach((incomingCase: CaseItem) => {
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
        setCases(sortedCases);

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
      let finalAiText = '';

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

  const getRelevance = (index: number) => {
    if (index === 0) return { label: 'High Match', color: '#059669', bg: '#D1FAE5' };
    if (index < 3) return { label: 'Strong Match', color: '#D97706', bg: '#FEF3C7' };
    return { label: 'Contextual', color: '#64748B', bg: '#F1F5F9' };
  };

  const renderGraphEdges = () => {
    const CENTER = 150;
    return (
      <Svg style={StyleSheet.absoluteFill}>
        {cases.map((c, index) => {
          const isCritical = (c.hitCount ?? 1) > 1;
          const radius = 60 + (index * 25);
          const angle = index * ((2 * Math.PI) / cases.length);
          const x = CENTER + radius * Math.cos(angle);
          const y = CENTER + radius * Math.sin(angle);
          const isSelected = selectedCase?.id === c.id;
          return (
            <Line
              key={c.id}
              x1={CENTER} y1={CENTER} x2={x} y2={y}
              stroke={isSelected ? "#007AFF" : (isCritical ? "#FF9500" : "#E5E5EA")}
              strokeWidth={isSelected || isCritical ? 2 : 1}
              strokeDasharray={isSelected || isCritical ? "" : "4,4"}
            />
          );
        })}
      </Svg>
    );
  };

  const renderGraphNodes = () => {
    const CENTER = 150;
    return cases.map((c, index) => {
      const isCritical = (c.hitCount ?? 1) > 1;
      const radius = 60 + (index * 25);
      const angle = index * ((2 * Math.PI) / cases.length);
      const x = CENTER + radius * Math.cos(angle);
      const y = CENTER + radius * Math.sin(angle);
      const isSelected = selectedCase?.id === c.id;
      const rel = getRelevance(index);
      return (
        <TouchableOpacity key={c.id} style={[styles.nodeWrapper, { left: x - 15, top: y - 15 }]} onPress={() => setSelectedCase(c)}>
          <View style={[styles.graphNode, { borderColor: isSelected ? '#007AFF' : (isCritical ? '#FF9500' : rel.color), transform: [{ scale: isCritical ? 1.2 : 1 }] }]}>
            <View style={[styles.nodeDot, { backgroundColor: isSelected ? '#007AFF' : (isCritical ? '#FF9500' : rel.color) }]} />
          </View>
          <View style={styles.nodeLabelContainer}>
            <Text style={[styles.nodeLabelText, isSelected && { color: '#007AFF' }]} numberOfLines={1}>{isCritical && "🔥 "}{c.id}</Text>
            <Text style={styles.nodeDistanceText}>{(c.hitCount ?? 0) > 1 ? `Hits: ${c.hitCount}` : `Dist: ${c.distance?.toFixed(2) ?? 'N/A'}`}</Text>
          </View>
        </TouchableOpacity>
      );
    });
  };

  // ========== LOBBY ==========
  if (currentScreen === 'lobby') {
    return (
      <View style={styles.lobbyWrapper}>
        <View style={styles.lobbyHeader}>
          <Ionicons name="scale" size={40} color="#1D1D1F" />
          <Text style={styles.lobbyTitle}>Legal Scribe</Text>
          <Text style={styles.lobbySubtitle}>Intelligent Analysis Workspace</Text>
        </View>
        <View style={styles.lobbyGrid}>
          <TouchableOpacity style={styles.newProjectCard} onPress={() => setCurrentScreen('config')}>
            <View style={styles.addCircle}><Ionicons name="add" size={30} color="#FFFFFF" /></View>
            <Text style={styles.cardTitle}>New Matter</Text>
            <Text style={styles.cardDesc}>Upload files and start a new analysis.</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.libraryManagerCard} onPress={() => setCurrentScreen('library_manager')}>
            <View style={[styles.addCircle, { backgroundColor: '#1D1D1F' }]}><Ionicons name="library" size={24} color="#FFFFFF" /></View>
            <Text style={styles.cardTitle}>Firm Libraries</Text>
            <Text style={styles.cardDesc}>Upload bulk documents to train global databases.</Text>
          </TouchableOpacity>
          {savedSessions.map((s: Session) => (
            <TouchableOpacity key={s.id} style={styles.projectCard} onPress={() => openProject(s)}>
              <Ionicons name="folder-open" size={24} color="#007AFF" />
              <Text style={styles.cardTitle} numberOfLines={1}>{s.name}</Text>
              <Text style={styles.cardDesc}>Last active: {new Date(s.created_at).toLocaleDateString()}</Text>
            </TouchableOpacity>
          ))}
        </View>
      </View>
    );
  }

  // ========== LIBRARY MANAGER ==========
  if (currentScreen === 'library_manager') {
    return (
      <View style={styles.lobbyWrapper}>
        <View style={styles.configContainer}>
          <TouchableOpacity style={styles.backLink} onPress={() => setCurrentScreen('lobby')}>
            <Ionicons name="chevron-back" size={16} color="#007AFF" />
            <Text style={styles.backLinkText}>Home</Text>
          </TouchableOpacity>
          <Text style={styles.configHeader}>Manage Firm Libraries</Text>
          <Text style={styles.label}>Active Firm Databases</Text>
          <View style={styles.activeLibrariesContainer}>
            {availableDatabases.length === 0 ? (
              <Text style={{ color: '#86868B', fontStyle: 'italic', marginBottom: 15 }}>No master libraries compiled yet.</Text>
            ) : (
              availableDatabases.map((db: string) => (
                <View key={db} style={styles.libraryBadge}>
                  <Ionicons name="server-outline" size={14} color="#007AFF" />
                  <Text style={styles.libraryBadgeText}>{db}</Text>
                </View>
              ))
            )}
          </View>
          <View style={styles.hDivider} />
          <Text style={styles.configHeader}>Compile New Library</Text>
          <Text style={styles.label}>Library Identifier (No spaces)</Text>
          <TextInput style={styles.textInput} placeholder="e.g., delaware_corporate_law" value={libraryName} onChangeText={setLibraryName} autoCapitalize="none" />
          <Text style={styles.label}>Bulk Document Upload</Text>
          <TouchableOpacity style={styles.uploadBtn} onPress={() => pickFile(setLibraryFiles, true)}>
            <Ionicons name="documents-outline" size={20} color="#1D1D1F" />
            <Text style={styles.uploadBtnText}>Select PDF, DOCX, JSON, or TXT</Text>
          </TouchableOpacity>
          <ScrollView style={{ maxHeight: 150, marginBottom: 10 }}>
            {libraryFiles.map((f, i) => (
              <View key={i} style={styles.fileItemRow}>
                <Text style={styles.fileItem}>📄 {f.name}</Text>
                <TouchableOpacity onPress={() => removeFile(i, setLibraryFiles)}>
                  <Ionicons name="close-circle" size={16} color="#8E8E93" />
                </TouchableOpacity>
              </View>
            ))}
          </ScrollView>
          {isCreatingLibrary && (
            <View style={styles.progressWrapper}>
              <View style={{ flexDirection: 'row', justifyContent: 'space-between', marginBottom: 5 }}>
                <Text style={{ fontSize: 13, color: '#1D1D1F', fontWeight: '500' }}>Indexing Vectors...</Text>
                <Text style={{ fontSize: 13, color: '#007AFF', fontWeight: '700' }}>{uploadProgress}%</Text>
              </View>
              <View style={styles.progressBarBg}>
                <View style={[styles.progressBarFill, { width: `${uploadProgress}%` as any }]} />
              </View>
            </View>
          )}
          <TouchableOpacity
            style={[styles.launchBtn, (!libraryName || libraryFiles.length === 0 || isCreatingLibrary) && { backgroundColor: '#E5E5EA' }]}
            onPress={createFirmLibrary}
            disabled={!libraryName || libraryFiles.length === 0 || isCreatingLibrary}
          >
            <Text style={[styles.launchBtnText, (!libraryName || libraryFiles.length === 0 || isCreatingLibrary) && { color: '#8E8E93' }]}>Compile Library</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  }

  // ========== CONFIG ==========
  if (currentScreen === 'config') {
    return (
      <View style={[styles.lobbyWrapper, { alignItems: 'center', justifyContent: 'center' }]}>
        <View style={styles.configContainer}>
          <TouchableOpacity style={styles.backLink} onPress={() => setCurrentScreen('lobby')}>
            <Ionicons name="chevron-back" size={16} color="#007AFF" />
            <Text style={styles.backLinkText}>Library</Text>
          </TouchableOpacity>
          <Text style={styles.configHeader}>New Legal Matter</Text>
          <TextInput style={styles.textInput} placeholder="Matter Name (e.g. Smith v. Jones)" value={configName} onChangeText={setConfigName} />
          <TextInput style={[styles.textInput, { height: 80 }]} multiline placeholder="Description..." value={configDesc} onChangeText={setConfigDesc} />
          <Text style={styles.label}>Active Firm Databases</Text>
          <ScrollView style={styles.dbListContainer}>
            {availableDatabases.length === 0 && <Text style={{ color: '#86868B', fontStyle: 'italic' }}>No master libraries found.</Text>}
            {availableDatabases.map(dbName => (
              <TouchableOpacity key={dbName} style={[styles.dbToggle, selectedDatabases.includes(dbName) && styles.dbToggleActive]} onPress={() => toggleDatabaseSelection(dbName)}>
                <Ionicons name={selectedDatabases.includes(dbName) ? "checkmark-circle" : "ellipse-outline"} size={20} color={selectedDatabases.includes(dbName) ? "#007AFF" : "#86868B"} />
                <Text style={[styles.dbToggleText, selectedDatabases.includes(dbName) && { color: '#007AFF', fontWeight: '600' }]}>{dbName}</Text>
              </TouchableOpacity>
            ))}
          </ScrollView>
          <Text style={styles.label}>Local Discovery Files (Private to this matter)</Text>
          <TouchableOpacity style={styles.uploadBtn} onPress={() => pickFile(setUploadedFiles, true)}>
            <Ionicons name="cloud-upload-outline" size={20} color="#1D1D1F" />
            <Text style={styles.uploadBtnText}>Upload PDF, DOCX, JSON, or TXT</Text>
          </TouchableOpacity>
          <ScrollView style={{ maxHeight: 100 }}>
            {uploadedFiles.map((f, i) => (
              <View key={i} style={styles.fileItemRow}>
                <Text style={styles.fileItem}>📎 {f.name}</Text>
                <TouchableOpacity onPress={() => removeFile(i, setUploadedFiles)}>
                  <Ionicons name="close-circle" size={16} color="#8E8E93" />
                </TouchableOpacity>
              </View>
            ))}
          </ScrollView>
          <TouchableOpacity style={[styles.launchBtn, !configName && { backgroundColor: '#E5E5EA' }]} onPress={createProject} disabled={!configName}>
            {loadingState === 'creating' ? <ActivityIndicator color="#FFF" /> : <Text style={styles.launchBtnText}>Initialize Workspace</Text>}
          </TouchableOpacity>
          {configError !== '' && <Text style={styles.configErrorText}>{configError}</Text>}
        </View>
      </View>
    );
  }

  // ========== WORKSPACE ==========
  return (
    <View style={styles.workspaceWrapper}>
      <View style={styles.sidePanel}>
        <View style={styles.sideTop}>
          <Ionicons name="scale" size={28} color="#FFFFFF" />
          <TouchableOpacity style={styles.homeBtn} onPress={() => setCurrentScreen('lobby')}>
            <Ionicons name="home-outline" size={20} color="#FFFFFF" />
          </TouchableOpacity>
        </View>
        <View style={styles.sideBottom}>
          <View style={styles.sessionIndicator} />
        </View>
      </View>

      <KeyboardAvoidingView behavior="padding" style={styles.mainContainer}>
        {/* PANE 1: CHAT */}
        <View style={styles.pane}>
          <View style={styles.paneHeader}><Text style={styles.paneTitle}>Matter Chat</Text></View>
          <ScrollView style={styles.chatScroll} ref={scrollViewRef} onContentSizeChange={() => scrollViewRef.current?.scrollToEnd()}>
            {messages.map((m: Message) => (
              <View key={m.id} style={[styles.bubble, m.role === 'user' ? styles.userBubble : styles.aiBubble]}>
                {m.role === 'user' ? (
                  <>
                    {m.files && m.files.length > 0 && (
                      <View style={styles.bubbleFilesContainer}>
                        {m.files.map((file, i) => (
                          <View key={i} style={styles.bubbleFileTag}>
                            <Ionicons name="document-attach" size={12} color="#FFF" />
                            <Text style={styles.bubbleFileText}>{file}</Text>
                          </View>
                        ))}
                      </View>
                    )}
                    <Text style={styles.userText}>{m.text}</Text>
                  </>
                ) : (
                  <>
                    <Markdown style={markdownStyles}>
                      {m.text + (loadingState === 'generating' && m.id === messages[messages.length - 1]?.id ? ' ▋' : '')}
                    </Markdown>
                    {m.graph_state && m.graph_state !== "[]" && (
                      <TouchableOpacity style={styles.checkpointBtn} onPress={() => setCases(JSON.parse(m.graph_state ?? '[]'))}>
                        <Ionicons name="git-branch-outline" size={12} color="#007AFF" />
                        <Text style={styles.checkpointText}>Restore Graph State</Text>
                      </TouchableOpacity>
                    )}
                  </>
                )}
              </View>
            ))}
            {loadingState !== '' && <ActivityIndicator style={{ marginTop: 10 }} color="#007AFF" />}
          </ScrollView>

          <View style={styles.inputArea}>
            {uploadedFiles.length > 0 && (
              <View style={styles.fileTagsContainer}>
                {uploadedFiles.map((file, i) => (
                  <View key={i} style={styles.fileTag}>
                    <Ionicons name="document-attach" size={12} color="#475569" />
                    <Text style={styles.fileTagText} numberOfLines={1}>{file.name}</Text>
                    <TouchableOpacity onPress={() => removeFile(i, setUploadedFiles)} style={styles.fileTagRemove} hitSlop={{ top: 6, bottom: 6, left: 6, right: 6 }}>
                      <Ionicons name="close" size={12} color="#94A3B8" />
                    </TouchableOpacity>
                  </View>
                ))}
              </View>
            )}
            <View style={styles.inputRow}>
              <TouchableOpacity style={styles.iconButton} onPress={() => pickFile(setUploadedFiles)}>
                <Ionicons name="add-circle" size={26} color="#64748B" />
              </TouchableOpacity>
              <TextInput style={styles.workspaceInput} placeholder="Query precedent or facts..." value={argument} onChangeText={setArgument} multiline />
              <TouchableOpacity
                style={[styles.sendBtn, (!argument.trim() && uploadedFiles.length === 0) && styles.sendBtnDisabled]}
                onPress={handleAnalyze}
                disabled={(!argument.trim() && uploadedFiles.length === 0) || loadingState !== ''}
              >
                <Ionicons name="arrow-up" size={20} color="#FFF" />
              </TouchableOpacity>
            </View>
          </View>
        </View>

        {/* PANE 2: GRAPH */}
        <View style={[styles.pane, { flex: 1.2 }]}>
          <View style={styles.paneHeader}><Text style={styles.paneTitle}>Semantic Network</Text></View>
          <View style={styles.graphBox}>
            <View style={{ transform: [{ scale: zoomScale }] }}>
              {cases.length > 0 ? (
                <>
                  {renderGraphEdges()}
                  {renderGraphNodes()}
                  <View style={styles.centerNode}><Text style={styles.centerNodeText}>Matter</Text></View>
                </>
              ) : <Text style={styles.emptyText}>Network will grow as you chat.</Text>}
            </View>
            <View style={styles.zoomBar}>
              <TouchableOpacity onPress={() => setCases([])}><Ionicons name="trash-outline" size={18} color="#8E8E93" /></TouchableOpacity>
              <View style={styles.vDivider} />
              <TouchableOpacity onPress={() => setZoomScale(z => z + 0.1)}><Text style={styles.zoomText}>+</Text></TouchableOpacity>
              <TouchableOpacity onPress={() => setZoomScale(z => z - 0.1)}><Text style={styles.zoomText}>-</Text></TouchableOpacity>
            </View>
          </View>
          <ScrollView style={{ flex: 1 }}>
            {cases.map((c: CaseItem, index: number) => {
              const rel = getRelevance(index);
              return (
                <TouchableOpacity key={c.id} style={[styles.caseRow, selectedCase?.id === c.id && styles.selectedRow]} onPress={() => setSelectedCase(c)}>
                  <Text style={styles.caseRowTitle}>{c.id}</Text>
                  <View style={{ flexDirection: 'row', alignItems: 'center', gap: 6 }}>
                    <View style={[styles.relBadge, { backgroundColor: rel.bg }]}>
                      <Text style={[styles.relBadgeText, { color: rel.color }]}>{rel.label}</Text>
                    </View>
                    {(c.hitCount ?? 1) > 1 && (
                      <View style={styles.hitCountBadge}><Text style={styles.hitCountText}>{c.hitCount ?? 1}</Text></View>
                    )}
                  </View>
                </TouchableOpacity>
              );
            })}
          </ScrollView>
        </View>

        {/* PANE 3: VIEWER */}
        <View style={styles.pane}>
          <View style={styles.paneHeader}><Text style={styles.paneTitle}>Source Document</Text></View>
          <ScrollView style={{ flex: 1 }}>
            {selectedCase ? (
              <>
                <Text style={styles.viewerTitle}>{selectedCase.id}</Text>
                <View style={styles.hDivider} />
                <Markdown style={markdownStyles}>{selectedCase.text}</Markdown>
              </>
            ) : <Text style={styles.emptyText}>Select a node to view source.</Text>}
          </ScrollView>
        </View>
      </KeyboardAvoidingView>
    </View>
  );
}

const styles = StyleSheet.create({
  lobbyWrapper: { flex: 1, backgroundColor: '#F5F5F7', padding: 60, alignItems: 'center' },
  lobbyHeader: { alignItems: 'center', marginBottom: 60 },
  lobbyTitle: { fontSize: 34, fontWeight: '700', color: '#1D1D1F', marginTop: 10 },
  lobbySubtitle: { fontSize: 17, color: '#86868B' },
  lobbyGrid: { flexDirection: 'row', flexWrap: 'wrap', justifyContent: 'center', gap: 20, width: '100%', maxWidth: 1000 },
  newProjectCard: { width: 220, height: 180, backgroundColor: '#FFFFFF', borderRadius: 20, padding: 24, alignItems: 'center', justifyContent: 'center', borderStyle: 'dashed', borderWidth: 2, borderColor: '#D2D2D7' },
  libraryManagerCard: { width: 220, height: 180, backgroundColor: '#E8F1FF', borderRadius: 20, padding: 24, alignItems: 'center', justifyContent: 'center', borderWidth: 2, borderColor: '#007AFF' },
  projectCard: { width: 220, height: 180, backgroundColor: '#FFFFFF', borderRadius: 20, padding: 24, shadowColor: '#000', shadowOpacity: 0.05, shadowRadius: 15 },
  addCircle: { width: 50, height: 50, borderRadius: 25, backgroundColor: '#007AFF', alignItems: 'center', justifyContent: 'center', marginBottom: 15 },
  cardTitle: { fontSize: 17, fontWeight: '600', color: '#1D1D1F', marginTop: 10 },
  cardDesc: { fontSize: 13, color: '#86868B', textAlign: 'center', marginTop: 5 },

  configContainer: { width: 500, backgroundColor: '#FFF', padding: 40, borderRadius: 30, shadowColor: '#000', shadowOpacity: 0.1, shadowRadius: 30 },
  backLink: { flexDirection: 'row', alignItems: 'center', marginBottom: 20 },
  backLinkText: { color: '#007AFF', marginLeft: 5, fontWeight: '600' },
  configHeader: { fontSize: 24, fontWeight: '700', marginBottom: 25, color: '#1D1D1F' },
  textInput: { backgroundColor: '#F5F5F7', padding: 15, borderRadius: 12, marginBottom: 15, fontSize: 16 },
  label: { fontSize: 14, fontWeight: '600', color: '#86868B', marginBottom: 10 },
  configErrorText: { color: '#EF4444', fontSize: 13, fontWeight: '500', marginTop: 12, textAlign: 'center' },
  activeLibrariesContainer: { flexDirection: 'row', flexWrap: 'wrap', marginBottom: 10 },
  libraryBadge: { flexDirection: 'row', alignItems: 'center', backgroundColor: '#F0F8FF', paddingHorizontal: 12, paddingVertical: 8, borderRadius: 10, marginRight: 10, marginBottom: 10, borderWidth: 1, borderColor: '#BFDBFE' },
  libraryBadgeText: { marginLeft: 6, color: '#007AFF', fontWeight: '600', fontSize: 13 },
  progressWrapper: { marginBottom: 15, padding: 15, backgroundColor: '#F5F5F7', borderRadius: 12 },
  progressBarBg: { height: 8, backgroundColor: '#E5E5EA', borderRadius: 4, overflow: 'hidden' },
  progressBarFill: { height: 8, backgroundColor: '#34C759' },
  dbListContainer: { maxHeight: 120, marginBottom: 20 },
  dbToggle: { flexDirection: 'row', alignItems: 'center', padding: 12, borderRadius: 10, backgroundColor: '#F5F5F7', marginBottom: 8, borderWidth: 1, borderColor: '#E5E5EA' },
  dbToggleActive: { backgroundColor: '#E8F1FF', borderColor: '#007AFF' },
  dbToggleText: { marginLeft: 10, fontSize: 15, color: '#1D1D1F' },
  uploadBtn: { flexDirection: 'row', alignItems: 'center', padding: 15, borderStyle: 'dashed', borderWidth: 1, borderColor: '#D2D2D7', borderRadius: 12, marginBottom: 10 },
  uploadBtnText: { marginLeft: 10, color: '#1D1D1F', fontWeight: '500' },
  fileItem: { fontSize: 13, color: '#007AFF', flex: 1 },
  fileItemRow: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginBottom: 5 },
  launchBtn: { backgroundColor: '#1D1D1F', padding: 18, borderRadius: 15, alignItems: 'center', marginTop: 20 },
  launchBtnText: { color: '#FFF', fontWeight: '700', fontSize: 16 },

  workspaceWrapper: { flex: 1, flexDirection: 'row', backgroundColor: '#F5F5F7' },
  sidePanel: { width: 70, backgroundColor: '#1D1D1F', alignItems: 'center', paddingVertical: 30, justifyContent: 'space-between' },
  sideTop: { alignItems: 'center', gap: 30 },
  sideBottom: { alignItems: 'center' },
  homeBtn: { padding: 10, backgroundColor: '#3A3A3C', borderRadius: 12 },
  sessionIndicator: { width: 8, height: 8, borderRadius: 4, backgroundColor: '#34C759' },
  mainContainer: { flex: 1, flexDirection: 'row', padding: 15, gap: 15 },
  pane: { flex: 1, backgroundColor: '#FFF', borderRadius: 24, padding: 20, shadowColor: '#000', shadowOpacity: 0.03, shadowRadius: 10 },
  paneHeader: { marginBottom: 15 },
  paneTitle: { fontSize: 15, fontWeight: '700', color: '#86868B', textTransform: 'uppercase', letterSpacing: 1 },

  chatScroll: { flex: 1 },
  bubble: { padding: 15, borderRadius: 18, marginBottom: 12, maxWidth: '90%' },
  userBubble: { backgroundColor: '#007AFF', alignSelf: 'flex-end' },
  aiBubble: { backgroundColor: '#F5F5F7', alignSelf: 'flex-start' },
  userText: { color: '#FFF', fontSize: 15 },
  bubbleFilesContainer: { flexDirection: 'row', flexWrap: 'wrap', marginBottom: 8 },
  bubbleFileTag: { flexDirection: 'row', alignItems: 'center', backgroundColor: 'rgba(255,255,255,0.25)', paddingHorizontal: 8, paddingVertical: 4, borderRadius: 6, marginRight: 6, marginBottom: 4 },
  bubbleFileText: { fontSize: 12, color: '#FFF', marginLeft: 4, fontWeight: '600' },
  checkpointBtn: { flexDirection: 'row', alignItems: 'center', marginTop: 10, padding: 6, backgroundColor: '#FFF', borderRadius: 8, alignSelf: 'flex-start' },
  checkpointText: { fontSize: 11, color: '#007AFF', marginLeft: 5, fontWeight: '600' },

  inputArea: { marginTop: 15 },
  fileTagsContainer: { flexDirection: 'row', flexWrap: 'wrap', marginBottom: 8 },
  fileTag: { flexDirection: 'row', alignItems: 'center', backgroundColor: '#F1F5F9', paddingHorizontal: 10, paddingVertical: 6, borderRadius: 8, marginRight: 8, marginBottom: 6, maxWidth: 200 },
  fileTagText: { fontSize: 13, color: '#334155', marginLeft: 6, fontWeight: '500', flex: 1 },
  fileTagRemove: { marginLeft: 6 },
  inputRow: { flexDirection: 'row', alignItems: 'flex-end', gap: 8 },
  iconButton: { padding: 8 },
  workspaceInput: { flex: 1, backgroundColor: '#F5F5F7', borderRadius: 15, padding: 12, maxHeight: 100, fontSize: 15 },
  sendBtn: { width: 44, height: 44, borderRadius: 22, backgroundColor: '#1D1D1F', alignItems: 'center', justifyContent: 'center' },
  sendBtnDisabled: { backgroundColor: '#E5E5EA' },

  graphBox: { height: 320, backgroundColor: '#F5F5F7', borderRadius: 20, alignItems: 'center', justifyContent: 'center', position: 'relative', overflow: 'hidden', marginBottom: 15 },
  centerNode: { position: 'absolute', width: 50, height: 50, borderRadius: 25, backgroundColor: '#1D1D1F', alignItems: 'center', justifyContent: 'center', top: 135, left: 135 },
  centerNodeText: { color: '#FFF', fontSize: 10, fontWeight: '700' },
  nodeWrapper: { position: 'absolute', width: 30, height: 30, alignItems: 'center', justifyContent: 'center' },
  graphNode: { width: 18, height: 18, borderRadius: 9, borderWidth: 2, backgroundColor: '#FFF', alignItems: 'center', justifyContent: 'center' },
  nodeDot: { width: 6, height: 6, borderRadius: 3 },
  nodeLabelContainer: { position: 'absolute', top: 22, width: 80, alignItems: 'center' },
  nodeLabelText: { fontSize: 9, fontWeight: '600', textAlign: 'center', color: '#1D1D1F' },
  nodeDistanceText: { fontSize: 8, color: '#86868B', textAlign: 'center', marginTop: 1 },
  zoomBar: { position: 'absolute', bottom: 10, right: 10, flexDirection: 'row', backgroundColor: '#FFF', borderRadius: 8, padding: 5, alignItems: 'center', gap: 8 },
  zoomText: { fontSize: 18, fontWeight: '500', color: '#1D1D1F' },
  vDivider: { width: 1, height: 15, backgroundColor: '#E5E5EA' },

  caseRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', padding: 12, borderBottomWidth: 1, borderBottomColor: '#F5F5F7' },
  selectedRow: { backgroundColor: '#F5F5F7' },
  caseRowTitle: { fontSize: 13, fontWeight: '500', color: '#1D1D1F', flex: 1, marginRight: 8 },
  relBadge: { paddingHorizontal: 8, paddingVertical: 4, borderRadius: 6 },
  relBadgeText: { fontSize: 10, fontWeight: '700' },
  hitCountBadge: { backgroundColor: '#FF9500', borderRadius: 10, paddingHorizontal: 6, paddingVertical: 2 },
  hitCountText: { color: '#FFF', fontSize: 10, fontWeight: '700' },

  viewerTitle: { fontSize: 20, fontWeight: '700', color: '#1D1D1F' },
  hDivider: { height: 1, backgroundColor: '#E5E5EA', marginVertical: 15 },
  emptyText: { color: '#86868B', textAlign: 'center', marginTop: 50 },
});

const markdownStyles: Record<string, TextStyle> = {
  body: { fontSize: 15, color: '#1D1D1F', lineHeight: 24 },
  heading1: { fontSize: 18, fontWeight: '700', marginVertical: 10 },
  heading2: { fontSize: 16, fontWeight: '600', marginVertical: 8 },
  strong: { fontWeight: '700' },
  blockquote: { borderLeftWidth: 3, borderLeftColor: '#007AFF', paddingLeft: 12, marginVertical: 8, backgroundColor: '#F5F5F7', paddingVertical: 8 },
};
