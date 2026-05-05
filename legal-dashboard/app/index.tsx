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
  graph_state?: string;
};

type PickerAsset = DocumentPicker.DocumentPickerAsset & { file?: File };

export default function Index() {
  // --- APP NAVIGATION ---
  const [currentScreen, setCurrentScreen] = useState<Screen>('lobby'); 
  
  // --- SESSION DATA ---
  const [savedSessions, setSavedSessions] = useState<Session[]>([]);
  const [availableDatabases, setAvailableDatabases] = useState<string[]>([]); 
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [sessionName, setSessionName] = useState('');
  
  // --- MATTER CONFIG STATE ---
  const [configName, setConfigName] = useState('');
  const [configDesc, setConfigDesc] = useState('');
  const [selectedDatabases, setSelectedDatabases] = useState<string[]>([]); 
  const [uploadedFiles, setUploadedFiles] = useState<PickerAsset[]>([]); 
  
  // --- LIBRARY MANAGER STATE (Super-User) ---
  const [libraryName, setLibraryName] = useState('');
  const [libraryFiles, setLibraryFiles] = useState<PickerAsset[]>([]);
  const [isCreatingLibrary, setIsCreatingLibrary] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0); // NEW: Progress State

  // --- WORKSPACE STATE ---
  const [messages, setMessages] = useState<Message[]>([]); 
  const [argument, setArgument] = useState('');
  const [loadingState, setLoadingState] = useState<LoadingState>(''); 
  const [cases, setCases] = useState<CaseItem[]>([]);
  const [selectedCase, setSelectedCase] = useState<CaseItem | null>(null);
  const [zoomScale, setZoomScale] = useState(1);
  const scrollViewRef = useRef<ScrollView | null>(null);

  // --- 1. INITIALIZATION LOGIC ---
  useEffect(() => {
    // UPDATED: Fetch databases when entering Library Manager as well
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

  // --- 2. UPLOAD HANDLERS ---
  const pickFile = async (
    setFilesState: React.Dispatch<React.SetStateAction<PickerAsset[]>>,
    multiple = false
  ) => {
    try {
      const result = await DocumentPicker.getDocumentAsync({ 
        type: [
          'application/pdf', 
          'text/plain',
          'application/json',
          'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        ], 
        copyToCacheDirectory: true,
        multiple: multiple
      });
      if (!result.canceled) {
        setFilesState(prev => [...prev, ...result.assets]);
      }
    } catch (error) { console.error("Picker error", error); }
  };

  const uploadFileToBackend = async (file: PickerAsset, sessionId: string) => {
    const formData = new FormData();
    formData.append('session_id', sessionId);
    const fileToUpload = Platform.OS === 'web' && file.file
      ? file.file
      : { uri: file.uri, name: file.name, type: file.mimeType || 'application/pdf' };
    formData.append('file', fileToUpload as any);
    await fetch(`${API_BASE}/upload`, { method: 'POST', body: formData });
  };

  const createFirmLibrary = async () => {
    if (!libraryName.trim() || libraryFiles.length === 0) return;
    setIsCreatingLibrary(true);
    setUploadProgress(0);
    
    try {
      const formData = new FormData();
      formData.append('db_name', libraryName);
      
      for (const file of libraryFiles) {
        // Check if web or mobile
        const fileToUpload = Platform.OS === 'web' 
          ? file.file 
          : { uri: file.uri, name: file.name, type: file.mimeType || 'application/pdf' };
          
        formData.append('files', fileToUpload as any);
      }
      
      const res = await fetch(`${API_BASE}/databases/create`, { method: 'POST', body: formData });
      
      // If FastAPI still throws an error, this will print the exact reason to your console!
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
  // --- 4. MATTER WORKFLOW ACTIONS ---
  const toggleDatabaseSelection = (dbName: string) => {
    setSelectedDatabases(prev => 
      prev.includes(dbName) ? prev.filter(db => db !== dbName) : [...prev, dbName]
    );
  };

  const createProject = async () => {
    if (!configName.trim()) return;
    setLoadingState('creating');
    try {
      const dbsToUse = [...selectedDatabases, "user_workspace"].join(",");

      const res = await fetch(`${API_BASE}/sessions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: configName, description: configDesc, databases: dbsToUse })
      });
      const data = await res.json();
      
      for (const file of uploadedFiles) await uploadFileToBackend(file, data.id);
      
      setActiveSessionId(data.id);
      setSessionName(data.name);
      setMessages([{ id: Date.now().toString(), role: 'assistant', text: `Workspace for **${data.name}** is ready. Analysis engine active.` }]);
      setCases([]);
      setCurrentScreen('workspace');
      setUploadedFiles([]);
      setConfigName('');
    } catch (error) { console.error("Creation error", error); }
    finally { setLoadingState(''); }
  };

  const openProject = async (session: Session) => {
    setActiveSessionId(session.id);
    setSessionName(session.name);
    setCurrentScreen('workspace');
    setCases([]);
    try {
      const res = await fetch(`${API_BASE}/sessions/${session.id}/messages`);
      const data = await res.json();
      if (data.messages?.length > 0) {
        setMessages(data.messages);
        const lastWithGraph = [...data.messages].reverse().find((m: Message) => m.graph_state && m.graph_state !== "[]");
        if (lastWithGraph) setCases(JSON.parse(lastWithGraph.graph_state));
      }
    } catch (error) { console.error("Load error", error); }
  };

  const handleAnalyze = async () => {
    if (!argument.trim()) return;
    const userText = argument;
    const userMsg: Message = { id: Date.now().toString(), role: 'user', text: userText };
    setMessages(prev => [...prev, userMsg]);
    setArgument('');
    setLoadingState('searching');

    try {
      const sRes = await fetch(`${API_BASE}/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: activeSessionId, argument: userText }),
      });
      const sData = await sRes.json();
      
      let currentGraphState: CaseItem[] = [];
      setCases(prev => {
        const updated = [...prev];
        sData.cases.forEach((inc: CaseItem) => {
          const idx = updated.findIndex(c => c.id === inc.id);
          if (idx >= 0) {
            updated[idx].hitCount = (updated[idx].hitCount || 1) + 1;
            updated[idx].distance = (updated[idx].distance + inc.distance) / 2;
          } else { updated.push({ ...inc, hitCount: 1 }); }
        });
        currentGraphState = updated.sort((a,b) => (b.hitCount||1) - (a.hitCount||1));
        return currentGraphState;
      });

      setLoadingState('generating');
      const aiId = Date.now().toString();
      setMessages(prev => [...prev, { id: aiId, role: 'assistant', text: '', graph_state: JSON.stringify(currentGraphState) }]);

      const gRes = await fetch(`${API_BASE}/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: activeSessionId, argument: userText, context_text: sData.context_text }),
      });

      if (!gRes.body) throw new Error('Missing response stream while generating memo');
      const reader = gRes.body.getReader();
      const decoder = new TextDecoder();
      let finishedText = "";
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value);
        finishedText += chunk;
        setMessages(prev => prev.map(m => m.id === aiId ? { ...m, text: finishedText } : m));
      }

      await fetch(`${API_BASE}/messages`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: activeSessionId, role: 'user', content: userText, graph_state: "[]" })
      });
      await fetch(`${API_BASE}/messages`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: activeSessionId, role: 'assistant', content: finishedText, graph_state: JSON.stringify(currentGraphState) })
      });

    } catch (e) { console.error(e); }
    finally { setLoadingState(''); }
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

      return (
        <React.Fragment key={c.id}>
          <Svg style={StyleSheet.absoluteFill}><Line x1={CENTER} y1={CENTER} x2={x} y2={y} stroke={isSelected ? "#007AFF" : (isCritical ? "#FF9500" : "#E5E5EA")} strokeWidth={isCritical || isSelected ? 2 : 1} strokeDasharray={isCritical || isSelected ? "" : "4,4"} /></Svg>
          <TouchableOpacity style={[styles.nodeWrapper, { left: x - 15, top: y - 15 }]} onPress={() => setSelectedCase(c)}>
            <View style={[styles.graphNode, { borderColor: isSelected ? '#007AFF' : (isCritical ? '#FF9500' : '#8E8E93'), transform: [{ scale: isCritical ? 1.2 : 1 }] }]}><View style={[styles.nodeDot, { backgroundColor: isSelected ? '#007AFF' : (isCritical ? '#FF9500' : '#8E8E93') }]} /></View>
            <View style={styles.nodeLabelContainer}><Text style={[styles.nodeLabelText, isSelected && {color: '#007AFF'}]} numberOfLines={1}>{isCritical && "🔥 "}{c.id}</Text></View>
          </TouchableOpacity>
        </React.Fragment>
      );
    });
  };

  // ==========================================
  // PAGE 1: THE LOBBY
  // ==========================================
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
            <View style={[styles.addCircle, {backgroundColor: '#1D1D1F'}]}><Ionicons name="library" size={24} color="#FFFFFF" /></View>
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

  // ==========================================
  // PAGE 2: LIBRARY MANAGER (Super User)
  // ==========================================
  if (currentScreen === 'library_manager') {
    return (
      <View style={styles.lobbyWrapper}>
        <View style={styles.configContainer}>
          <TouchableOpacity style={styles.backLink} onPress={() => setCurrentScreen('lobby')}>
            <Ionicons name="chevron-back" size={16} color="#007AFF" />
            <Text style={styles.backLinkText}>Home</Text>
          </TouchableOpacity>
          <Text style={styles.configHeader}>Manage Firm Libraries</Text>
          
          {/* --- NEW: DISPLAY EXISTING LIBRARIES --- */}
          <Text style={styles.label}>Active Firm Databases</Text>
          <View style={styles.activeLibrariesContainer}>
            {availableDatabases.length === 0 ? (
                <Text style={{color: '#86868B', fontStyle: 'italic', marginBottom: 15}}>No master libraries compiled yet.</Text>
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
          
          <ScrollView style={{maxHeight: 150, marginBottom: 10}}>
            {libraryFiles.map((f, i) => <Text key={i} style={styles.fileItem}>📄 {f.name}</Text>)}
          </ScrollView>

          {/* --- NEW: PROGRESS BAR --- */}
          {isCreatingLibrary && (
             <View style={styles.progressWrapper}>
                <View style={{flexDirection: 'row', justifyContent: 'space-between', marginBottom: 5}}>
                   <Text style={{fontSize: 13, color: '#1D1D1F', fontWeight: '500'}}>Indexing Vectors...</Text>
                   <Text style={{fontSize: 13, color: '#007AFF', fontWeight: '700'}}>{uploadProgress}%</Text>
                </View>
                <View style={styles.progressBarBg}>
                   <View style={[styles.progressBarFill, { width: `${uploadProgress}%` }]} />
                </View>
             </View>
          )}

          <TouchableOpacity style={[styles.launchBtn, (!libraryName || libraryFiles.length === 0 || isCreatingLibrary) && {backgroundColor: '#E5E5EA'}]} onPress={createFirmLibrary} disabled={!libraryName || libraryFiles.length === 0 || isCreatingLibrary}>
             <Text style={[styles.launchBtnText, (!libraryName || libraryFiles.length === 0 || isCreatingLibrary) && {color: '#8E8E93'}]}>Compile Library</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  }

  // ==========================================
  // PAGE 3: CONFIGURATION (Normal User)
  // ==========================================
  if (currentScreen === 'config') {
    return (
      <View style={styles.lobbyWrapper}>
        <View style={styles.configContainer}>
          <TouchableOpacity style={styles.backLink} onPress={() => setCurrentScreen('lobby')}>
            <Ionicons name="chevron-back" size={16} color="#007AFF" />
            <Text style={styles.backLinkText}>Library</Text>
          </TouchableOpacity>
          <Text style={styles.configHeader}>New Legal Matter</Text>
          
          <TextInput style={styles.textInput} placeholder="Matter Name (e.g. Smith v. Jones)" value={configName} onChangeText={setConfigName} />
          <TextInput style={[styles.textInput, {height: 80}]} multiline placeholder="Description..." value={configDesc} onChangeText={setConfigDesc} />
          
          <Text style={styles.label}>Active Firm Databases</Text>
          <ScrollView style={styles.dbListContainer}>
            {availableDatabases.length === 0 && <Text style={{color: '#86868B', fontStyle: 'italic'}}>No master libraries found.</Text>}
            {availableDatabases.map(dbName => (
              <TouchableOpacity key={dbName} style={[styles.dbToggle, selectedDatabases.includes(dbName) && styles.dbToggleActive]} onPress={() => toggleDatabaseSelection(dbName)}>
                <Ionicons name={selectedDatabases.includes(dbName) ? "checkmark-circle" : "ellipse-outline"} size={20} color={selectedDatabases.includes(dbName) ? "#007AFF" : "#86868B"} />
                <Text style={[styles.dbToggleText, selectedDatabases.includes(dbName) && {color: '#007AFF', fontWeight: '600'}]}>{dbName}</Text>
              </TouchableOpacity>
            ))}
          </ScrollView>
          
          <Text style={styles.label}>Local Discovery Files (Private to this matter)</Text>
          <TouchableOpacity style={styles.uploadBtn} onPress={() => pickFile(setUploadedFiles, true)}>
            <Ionicons name="cloud-upload-outline" size={20} color="#1D1D1F" />
            <Text style={styles.uploadBtnText}>Upload PDF, DOCX, JSON, or TXT</Text>
          </TouchableOpacity>
          
          <ScrollView style={{maxHeight: 100}}>
             {uploadedFiles.map((f, i) => <Text key={i} style={styles.fileItem}>📎 {f.name}</Text>)}
          </ScrollView>

          <TouchableOpacity style={[styles.launchBtn, !configName && {backgroundColor: '#E5E5EA'}]} onPress={createProject} disabled={!configName}>
            {loadingState === 'creating' ? <ActivityIndicator color="#FFF" /> : <Text style={styles.launchBtnText}>Initialize Workspace</Text>}
          </TouchableOpacity>
        </View>
      </View>
    );
  }

  // ==========================================
  // PAGE 4: THE WORKSPACE
  // ==========================================
  return (
    <View style={styles.workspaceWrapper}>
      <View style={styles.sidePanel}>
        <View style={styles.sideTop}>
          <Ionicons name="scale" size={28} color="#FFFFFF" />
          <TouchableOpacity style={styles.homeBtn} onPress={() => setCurrentScreen('lobby')}>
            <Ionicons name="home-outline" size={20} color="#FFFFFF" />
          </TouchableOpacity>
        </View>
        <View style={styles.sideBottom}><View style={styles.sessionIndicator} /></View>
      </View>

      <KeyboardAvoidingView behavior="padding" style={styles.mainContainer}>
        {/* PANE 1: CHAT */}
        <View style={styles.pane}>
          <View style={styles.paneHeader}><Text style={styles.paneTitle}>Matter Chat</Text></View>
          <ScrollView style={styles.chatScroll} ref={scrollViewRef} onContentSizeChange={() => scrollViewRef.current?.scrollToEnd()}>
            {messages.map((m: Message) => (
              <View key={m.id} style={[styles.bubble, m.role === 'user' ? styles.userBubble : styles.aiBubble]}>
                {m.role === 'assistant' ? (
                   <>
                    <Markdown style={markdownStyles}>{m.text}</Markdown>
                    {m.graph_state && m.graph_state !== "[]" && (
                      <TouchableOpacity style={styles.checkpointBtn} onPress={() => setCases(JSON.parse(m.graph_state ?? '[]'))}>
                        <Ionicons name="git-branch-outline" size={12} color="#007AFF" />
                        <Text style={styles.checkpointText}>Restore Graph State</Text>
                      </TouchableOpacity>
                    )}
                   </>
                ) : <Text style={styles.userText}>{m.text}</Text>}
              </View>
            ))}
            {loadingState !== '' && <ActivityIndicator style={{marginTop: 10}} color="#007AFF" />}
          </ScrollView>
          <View style={styles.inputArea}>
             <TextInput style={styles.workspaceInput} placeholder="Query precedent or facts..." value={argument} onChangeText={setArgument} multiline />
             <TouchableOpacity style={styles.sendBtn} onPress={handleAnalyze}><Ionicons name="arrow-up" size={20} color="#FFF" /></TouchableOpacity>
          </View>
        </View>

        {/* PANE 2: GRAPH */}
        <View style={[styles.pane, {flex: 1.2}]}>
           <View style={styles.paneHeader}><Text style={styles.paneTitle}>Semantic Network</Text></View>
           <View style={styles.graphBox}>
              <View style={{ transform: [{ scale: zoomScale }] }}>
                {cases.length > 0 ? (
                  <>
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
           <ScrollView style={{flex: 1}}>
              {cases.map((c: CaseItem) => (
                <TouchableOpacity key={c.id} style={[styles.caseRow, selectedCase?.id === c.id && styles.selectedRow]} onPress={() => setSelectedCase(c)}>
                  <Text style={styles.caseRowTitle}>{c.id}</Text>
                  {(c.hitCount ?? 1) > 1 && <View style={styles.hitBadge}><Text style={styles.hitText}>{c.hitCount ?? 1}</Text></View>}
                </TouchableOpacity>
              ))}
           </ScrollView>
        </View>

        {/* PANE 3: VIEWER */}
        <View style={styles.pane}>
           <View style={styles.paneHeader}><Text style={styles.paneTitle}>Source Document</Text></View>
           <ScrollView style={{flex: 1}}>
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
  configHeader: { fontSize: 24, fontWeight: '700', marginBottom: 25 },
  textInput: { backgroundColor: '#F5F5F7', padding: 15, borderRadius: 12, marginBottom: 15, fontSize: 16 },
  label: { fontSize: 14, fontWeight: '600', color: '#86868B', marginBottom: 10 },
  
  // Library Manager Specifics
  activeLibrariesContainer: { flexDirection: 'row', flexWrap: 'wrap', marginBottom: 10 },
  libraryBadge: { flexDirection: 'row', alignItems: 'center', backgroundColor: '#F0F8FF', paddingHorizontal: 12, paddingVertical: 8, borderRadius: 10, marginRight: 10, marginBottom: 10, borderWidth: 1, borderColor: '#BFDBFE' },
  libraryBadgeText: { marginLeft: 6, color: '#007AFF', fontWeight: '600', fontSize: 13 },
  progressWrapper: { marginBottom: 15, padding: 15, backgroundColor: '#F5F5F7', borderRadius: 12 },
  progressBarBg: { height: 8, backgroundColor: '#E5E5EA', borderRadius: 4, overflow: 'hidden' },
  progressBarFill: { height: '100%', backgroundColor: '#34C759' }, // Apple Green

  dbListContainer: { maxHeight: 120, marginBottom: 20 },
  dbToggle: { flexDirection: 'row', alignItems: 'center', padding: 12, borderRadius: 10, backgroundColor: '#F5F5F7', marginBottom: 8, borderWidth: 1, borderColor: '#E5E5EA' },
  dbToggleActive: { backgroundColor: '#E8F1FF', borderColor: '#007AFF' },
  dbToggleText: { marginLeft: 10, fontSize: 15, color: '#1D1D1F' },

  uploadBtn: { flexDirection: 'row', alignItems: 'center', padding: 15, borderStyle: 'dashed', borderWidth: 1, borderColor: '#D2D2D7', borderRadius: 12, marginBottom: 10 },
  uploadBtnText: { marginLeft: 10, color: '#1D1D1F', fontWeight: '500' },
  fileItem: { fontSize: 13, color: '#007AFF', marginBottom: 5 },
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
  inputArea: { flexDirection: 'row', gap: 10, marginTop: 15 },
  workspaceInput: { flex: 1, backgroundColor: '#F5F5F7', borderRadius: 15, padding: 12, maxHeight: 100 },
  sendBtn: { width: 44, height: 44, borderRadius: 22, backgroundColor: '#1D1D1F', alignItems: 'center', justifyContent: 'center' },
  checkpointBtn: { flexDirection: 'row', alignItems: 'center', marginTop: 10, padding: 6, backgroundColor: '#FFF', borderRadius: 8, alignSelf: 'flex-start' },
  checkpointText: { fontSize: 11, color: '#007AFF', marginLeft: 5, fontWeight: '600' },

  graphBox: { height: 320, backgroundColor: '#F5F5F7', borderRadius: 20, alignItems: 'center', justifyContent: 'center', position: 'relative', overflow: 'hidden', marginBottom: 15 },
  centerNode: { position: 'absolute', width: 50, height: 50, borderRadius: 25, backgroundColor: '#1D1D1F', alignItems: 'center', justifyContent: 'center', top: 135, left: 135 },
  centerNodeText: { color: '#FFF', fontSize: 10, fontWeight: '700' },
  nodeWrapper: { position: 'absolute', width: 30, height: 30, alignItems: 'center', justifyContent: 'center' },
  graphNode: { width: 18, height: 18, borderRadius: 9, borderWidth: 2, backgroundColor: '#FFF', alignItems: 'center', justifyContent: 'center' },
  nodeDot: { width: 6, height: 6, borderRadius: 3 },
  nodeLabelContainer: { position: 'absolute', top: 22, width: 80, alignItems: 'center' },
  nodeLabelText: { fontSize: 9, fontWeight: '600', textAlign: 'center', color: '#1D1D1F' },
  zoomBar: { position: 'absolute', bottom: 10, right: 10, flexDirection: 'row', backgroundColor: '#FFF', borderRadius: 8, padding: 5, alignItems: 'center', gap: 8 },
  zoomText: { fontSize: 18, fontWeight: '500' },
  vDivider: { width: 1, height: 15, backgroundColor: '#E5E5EA' },

  caseRow: { flexDirection: 'row', justifyContent: 'space-between', padding: 12, borderBottomWidth: 1, borderBottomColor: '#F5F5F7' },
  selectedRow: { backgroundColor: '#F5F5F7' },
  caseRowTitle: { fontSize: 13, fontWeight: '500', color: '#1D1D1F', flex: 1 },
  hitBadge: { backgroundColor: '#FF9500', borderRadius: 10, paddingHorizontal: 6 },
  hitText: { color: '#FFF', fontSize: 10, fontWeight: '700' },
  viewerTitle: { fontSize: 20, fontWeight: '700', color: '#1D1D1F' },
  hDivider: { height: 1, backgroundColor: '#E5E5EA', marginVertical: 15 },
  emptyText: { color: '#86868B', textAlign: 'center', marginTop: 50 }
});

const markdownStyles: Record<string, TextStyle> = {
  body: { fontSize: 15, color: '#1D1D1F', lineHeight: 24 },
  heading1: { fontSize: 18, fontWeight: '700', marginVertical: 10 },
  strong: { fontWeight: '700' }
};